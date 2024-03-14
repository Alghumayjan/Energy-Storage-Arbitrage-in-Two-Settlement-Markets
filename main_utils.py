import os
import pandas as pd
import cvxpy as cp
import numpy as np

def run_script(script_dir, model, zone, log):
    if log:
        script_path = 'scripts/log_'+model+'_'+zone+'.sh'
    else:
        script_path = 'scripts/'+model+'_'+zone+'.sh'

    current_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(script_dir))
        os.system(f"bash {script_path}")
    finally:
        os.chdir(current_dir)

def pred_metrics(pred, true):
    pred_df = pd.DataFrame(pred[:,:,0])
    true_df = pd.DataFrame(true[:,:,0])
    rtp_mean = np.array(true_df[7*24::24]).flatten().mean()
    MAE = np.abs(np.array(true_df[7*24::24]).flatten() - np.array(pred_df[7*24::24]).flatten()).mean() 
    MAE_naive =  np.abs(np.array(true_df[7*24::24]).flatten() - np.array(true_df[:-7*24:24]).flatten()).mean()
    rMAE = MAE / MAE_naive
    return MAE, rMAE, rtp_mean

def DAM_Arb(DAP, RTP_1h_actual, RTP_1h, Power_rating, Energy_capacity,
                               Self_discharge_efficiency, Charging_Discharging_Efficiency,
                               State_of_charge_init, State_of_charge_final, State_of_charge_min,
                               State_of_charge_max, cost_discharge):
    days = 365
    T = 24
    DAP = np.array(DAP).reshape(days,T)
    DAP = np.array([np.concatenate((row[1:], [DAP[i+1][0]] if i < len(DAP) - 1 else [])) for i, row in enumerate(DAP[:-1])])

    RTP_1h_actual = np.array(RTP_1h_actual)
    RTP_1h = np.array(RTP_1h)
    Bdis = RTP_1h[:]  
    Bchr = RTP_1h[:]  

    # Variables
    s = cp.Variable(T + 1, nonneg=True)
    q_r = cp.Variable(T, nonneg=True)
    q_d = cp.Variable(T, nonneg=True)
    u = cp.Variable(T, boolean=True)

    # Constraints
    constraints = [
        s[1:] == Self_discharge_efficiency * s[:-1] + Charging_Discharging_Efficiency * q_r - q_d / Charging_Discharging_Efficiency,
        q_r <= Power_rating * u,
        q_d <= Power_rating * (1 - u),
        s[0] == State_of_charge_init * Energy_capacity,
        s[-1] >= State_of_charge_final * Energy_capacity,
        s >= State_of_charge_min * Energy_capacity,
        s <= State_of_charge_max * Energy_capacity
    ]

    # Objective function
    daily_profits = []
    C_d = np.zeros((days-1, T))  
    D_d = np.zeros((days-1, T))  
    SOC = np.zeros((days-1, T))  
    for dd in range(days-1):
        objective = cp.Maximize(cp.sum((DAP[dd, :] - Bdis[dd, :]) @ q_d + (Bchr[dd, :] - DAP[dd, :]) @ q_r - cost_discharge * (q_d + q_r)))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Extracting results
        C_d[dd, :] = q_r.value
        D_d[dd, :] = q_d.value
        SOC[dd, :] = s.value[1:]

        # Record daily profit
        daily_profits.append((DAP[dd, :] - RTP_1h_actual[dd, :]) @ D_d[dd, :] + (RTP_1h_actual[dd, :] - DAP[dd, :]) @ C_d[dd, :] - cost_discharge * sum(D_d[dd, :] + C_d[dd, :]))
        
        if dd % 30 == 0:
            print(f"Day {dd+1}, Cummulative Profits {sum(daily_profits)}")

    # Record the results
    RTP_1h_v = RTP_1h.reshape(RTP_1h.size,1).flatten()
    RTP_1h_actual_v = RTP_1h_actual.reshape(RTP_1h_actual.size,1).flatten()
    DAP_v = DAP.reshape(DAP.size,1).flatten()
    D_d_v = D_d.reshape(D_d.size,1).flatten()
    C_d_v = C_d.reshape(C_d.size,1).flatten()
    SOC_d_v = SOC.reshape(SOC.size,1).flatten()
    Rev_1h_v = (DAP_v-RTP_1h_actual_v)*(D_d_v-C_d_v)

    # Create a list of dictionaries representing the solution values
    solutions = {'DAP': DAP_v, 'RTP_forecasted': RTP_1h_v, 'RTP_actual': RTP_1h_actual_v,'Discharge': D_d_v, 'Charge': C_d_v, 'SoC': SOC_d_v, 'Revenue': Rev_1h_v}

    return pd.DataFrame(solutions)

def Arb_comb_profit(DAM_sol, RTM_sol):
    DAM_5m = DAM_sol[:-1].reindex(DAM_sol[:-1].index.repeat(12)).reset_index(drop=True)
    comb_5m_profit = (DAM_5m.DAP*(DAM_5m.Discharge/12 - DAM_5m.Charge/12) + RTM_sol.RTP*(RTM_sol.Discharge-RTM_sol.Charge - (DAM_5m.Discharge/12 - DAM_5m.Charge/12)) - 10*RTM_sol.Discharge)
    comb_profit = np.array([sum(comb_5m_profit[i:i+12*24]) for i in range(0, len(comb_5m_profit), 12*24)])
    comb_total_profit = np.sum(comb_profit)

    _, RTM_total_profit = Arb_RTM_profit(RTM_sol)

    IPM = comb_total_profit/RTM_total_profit*100 - 100

    VB_profit_h = (DAM_sol[:-1].DAP-DAM_sol[:-1].RTP_actual)*(DAM_sol[:-1].Discharge-DAM_sol[:-1].Charge)
    VB_profit = np.array([sum(VB_profit_h[i:i+24]) for i in range(0, len(VB_profit_h), 24)])

    return comb_profit, comb_total_profit, IPM, VB_profit

def Arb_RTM_profit(RTM_sol):
    RTM_5m_profit = (RTM_sol.RTP*(RTM_sol.Discharge-RTM_sol.Charge) - 10*RTM_sol.Discharge)
    RTM_profit = np.array([sum(RTM_5m_profit[i:i+12*24]) for i in range(0, len(RTM_5m_profit), 12*24)])
    RTM_total_profit = RTM_profit.sum()
    
    return RTM_profit, RTM_total_profit

def neg_profit(VB, RTM, comb):
    VB_neg = sum(profit < 0 for profit in VB)
    RTM_neg = sum(profit < 0 for profit in RTM)
    comb_neg = sum(profit < 0 for profit in comb)
    return VB_neg, RTM_neg, comb_neg


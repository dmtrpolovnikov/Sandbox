import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import seaborn as sns
import datetime
from dateutil.relativedelta import *

def make_sensitivity_matrix(
    final_spots : int = 95,
    libor_val: float = 5.6,
    percentage_var: int = 10,
    square_grid: int = 5,
    interest_rate: float = 15.6,
    
) -> None :
    
    spot_usd_rub = np.linspace(final_spots*(100 - percentage_var)/100, final_spots*(100 + percentage_var)/100, square_grid)
    libor_ir = np.linspace(libor_val*(100 - percentage_var)/100, libor_val*(100 + percentage_var)/100, square_grid)

    sensitivity_matrix = np.zeros((len(spot_usd_rub), len(libor_ir)), dtype=int)

    inner_libor = {}
    for i, inner_spot in enumerate(spot_usd_rub):
        for j, libor in enumerate(libor_ir):
            inner_libor[start_date] = libor
            sensitivity_matrix[i, j] = int( 
                construct_credit_swap(
                    volume=volume,

                    start_swap_date=start_date,
                    end_swap_date=end_date,

                    interest_rate=interest_rate, # in percentage i.e. 15.6%
                    libor=inner_libor,

                    current_spot=current_spot, 
                    final_spot=inner_spot, 

                    ir_rub_est=ir_rub_est,
                    ir_usd_est=ir_usd_est,

                    first_leg_payments_periodity=first_leg_payments_periodity,
                    second_leg_payments_periodity=second_leg_payments_periodity,

                    margin_bps=margin_bps,
                    render=False
                )
            )

    sensitivity_matrix = pd.DataFrame(sensitivity_matrix, columns=[spot_usd_rub], index = [libor_ir])
        
    plt.figure(figsize=(10, 7))

    cmap_1 = sns.light_palette("seagreen", as_cmap=True)
    cmap_2 = sns.color_palette("Reds_r", as_cmap=True)

    sns.heatmap(sensitivity_matrix[sensitivity_matrix<0], cmap=cmap_2, annot=True, fmt=".0f")
    sns.heatmap(sensitivity_matrix[sensitivity_matrix>0], cmap=cmap_1, annot=True, fmt=".0f")

    plt.xlabel('SPOT USD-RUB')
    plt.ylabel('LIBOR USD')
    
    
def construct_credit_swap(
    volume: int,
    start_swap_date: pd.Timestamp,
    end_swap_date: pd.Timestamp,
    interest_rate: float,
    libor: np.ndarray,
    current_spot: float,
    final_spot: float,
    ir_rub_est: np.ndarray,
    ir_usd_est: np.ndarray,
    
    
    first_leg_payments_periodity: int = 3,
    second_leg_payments_periodity: int = 6,
    margin_bps: int = 250,
    render: bool = True,
    
) -> float:
    
    global credit_swaps
    
    # input data
    volume = volume # USD
    spot_usd_rub = current_spot
    
    start_date = start_swap_date
    end_date = end_swap_date

    margin_bps = margin_bps

    # 1st leg info
    payment_direction = -1 
    multiplicator = -1 # we pay
    first_leg_payments_periodity = first_leg_payments_periodity
    basis = 'FIX'
    interest_rate_1st_leg = interest_rate / 100
    payment_per_period_1 = volume * spot_usd_rub * interest_rate_1st_leg / 12 * first_leg_payments_periodity * multiplicator
    denomination_payment = volume * spot_usd_rub * multiplicator

    # 2nd leg info
    payment_direction = -1
    multiplicator = 1 # we recieve
    second_leg_payments_periodity = second_leg_payments_periodity
    basis = 'LIBOR'

    credit_swaps = pd.DataFrame()
    month_line = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
    credit_swaps.index = [start_date + relativedelta(months=+i) for i in range(month_line)]
    credit_swaps.index = pd.to_datetime(credit_swaps.index)
    
    credit_swaps['LIBOR'] = np.zeros(len(credit_swaps.index))
    credit_swaps['Year IR RUB'] = np.zeros(len(credit_swaps.index))
    credit_swaps['Year IR USD'] = np.zeros(len(credit_swaps.index))
    
    if len(list(libor.keys())) > 1:
        for date in list(libor.keys())[:-1]:
            d_date = pd.to_datetime(date)
            credit_swaps['LIBOR'][credit_swaps['LIBOR'].index >= d_date] = libor[date]
    else:
        for date in list(libor.keys()):
            d_date = pd.to_datetime(date)
            credit_swaps['LIBOR'][credit_swaps['LIBOR'].index >= d_date] = libor[date]
        
        
    if len(list(ir_rub_est.keys())) > 1:
        for date in list(ir_rub_est.keys())[:-1]:
            d_date = pd.to_datetime(date)
            credit_swaps['Year IR RUB'][credit_swaps['LIBOR'].index >= d_date] = ir_rub_est[date]
    else:
        for date in list(ir_rub_est.keys()):
            d_date = pd.to_datetime(date)
            credit_swaps['Year IR RUB'][credit_swaps['LIBOR'].index >= d_date] = ir_rub_est[date]
    
    if len(list(ir_usd_est.keys())) > 1:
        for date in list(ir_usd_est.keys())[:-1]:
            d_date = pd.to_datetime(date)
            credit_swaps['Year IR USD'][credit_swaps['LIBOR'].index >= d_date] = ir_usd_est[date]
    else:
        for date in list(ir_usd_est.keys()):
            d_date = pd.to_datetime(date)
            credit_swaps['Year IR USD'][credit_swaps['LIBOR'].index >= d_date] = ir_usd_est[date]

    credit_swaps['Effective RUB'] = (1 + credit_swaps['Year IR RUB']/100) ** (1/12) - 1
    credit_swaps['Effective USD'] = (1 + credit_swaps['Year IR USD']/100) ** (1/12) - 1

    credit_swaps['1st_leg_flag'] = [1 if (i % first_leg_payments_periodity) == 0 else 0 for i, date in enumerate(credit_swaps.index)]
    credit_swaps['2nd_leg_flag'] = [1 if (i % second_leg_payments_periodity) == 0 else 0 for i, date in enumerate(credit_swaps.index)]

    credit_swaps['1st_cashflow (RUB)'] = credit_swaps['1st_leg_flag'].apply(lambda x: payment_per_period_1 if x == 1 else 0)
    credit_swaps['2nd_cashflow (USD)'] = (credit_swaps['LIBOR'] + margin_bps/100/100) * volume / 12 * second_leg_payments_periodity * multiplicator * credit_swaps['2nd_leg_flag']

    credit_swaps['1st_cashflow (RUB)'][-1] = denomination_payment
    credit_swaps['2nd_cashflow (USD)'][-1] = volume

    # credit_swaps['Full discount factor RUB'] = 1/(1+credit_swaps['Effective RUB'][0])
    credit_swaps['Discount factor RUB'] = [1/(1 + credit_swaps['Effective RUB'][i]) ** (i+1) for i, date in enumerate(credit_swaps.index)] 
    credit_swaps['Discount factor USD'] = [1/(1 + credit_swaps['Effective USD'][i]) ** (i+1) for i, date in enumerate(credit_swaps.index)] 

    credit_swaps['Discounted cashflow RUB'] = credit_swaps['1st_cashflow (RUB)'] * credit_swaps['Discount factor RUB']
    credit_swaps['Discounted cashflow USD'] = credit_swaps['2nd_cashflow (USD)'] * credit_swaps['Discount factor USD']

    credit_swaps['Discounted cashflow RUB'][-1] = denomination_payment * credit_swaps['Discount factor RUB'][-1]
    credit_swaps['Discounted cashflow USD'][-1] = volume * credit_swaps['Discount factor USD'][-1]

    credit_swaps = credit_swaps.drop(columns=['1st_leg_flag', '2nd_leg_flag'])
    # credit_swaps = credit_swaps.iloc[:,1:].loc[~(credit_swaps.iloc[:,1:]==0).all(axis=1)]
    
    discounted_sum_leg_1 = credit_swaps['Discounted cashflow RUB'].sum()
    discounted_sum_leg_2 = credit_swaps['Discounted cashflow USD'].sum()
    
    if render:
        
        plt.figure(figsize=(15,6))
        
        plt.plot(credit_swaps['Year IR RUB'], label='RUB year interest rate')
        plt.plot(credit_swaps['Year IR USD'], label='USD year interest rate')
        plt.plot(credit_swaps['LIBOR']*100, label='USD LIBOR')
        
        plt.legend()
        plt.grid()
        
        plt.show()

        plt.figure(figsize=(15,6))

        plt.bar(np.arange(1, len(credit_swaps)), height=credit_swaps['Discounted cashflow USD'][:-1], label='USD cashflow')
        plt.bar(np.arange(1, len(credit_swaps)), height=credit_swaps['Discounted cashflow RUB'][:-1]/current_spot, label=f'RUB cashflow scaled to USD on current spot {current_spot}')

        plt.legend()
        plt.title('Interest cashflow')
        plt.ylabel('Money amount')
        plt.grid()
        plt.show()
    
    #print(f'inner = {final_spot}, libor = {libor}')
    
    return discounted_sum_leg_1 / final_spot + discounted_sum_leg_2


def start_calculator(make_sensitivity_mx=False):
  
    global volume
    global interest_rate
    global current_spot
    global final_spot
    global first_leg_payments_periodity
    global second_leg_payments_periodity
    global margin_bps
    global start_date
    global end_date
    global ir_rub_est
    global ir_usd_est
    # global libor
    
    print('volume')
    volume = int(input())
    
    print("interest_rate")
    interest_rate = float(input())
    
    print("current_spot")
    current_spot = float(input())
    final_spot = current_spot
    
    print("first_leg_payments_periodity, second_leg_payments_periodity")    
    first_leg_payments_periodity, second_leg_payments_periodity = [int(num) for num in input().split(sep=' ')]
    
    print("margin_bps")
    margin_bps = int(input())

    print("enter start, end dates in format YYYY-MM-DD: ")
    start_date, end_date = [pd.to_datetime(a) for a in input().split(sep=' ')]
    n_array = (end_date.year - start_date.year)*12 + (end_date.month - start_date.month)

    ir_rub_est = {}
    ir_usd_est = {}
    libor = {}

    print("ir rub est. input in format 'YYYY-MM-DD' value")
    date = start_date
    while date < end_date:
        date, value = input().split(sep=' ')
        date = pd.to_datetime(date)
        ir_rub_est[date] = float(value)
        
    print("ir usd est. input in format 'YYYY-MM-DD' value")
    date = start_date
    while date < end_date:
        date, value = input().split(sep=' ')
        date = pd.to_datetime(date)
        ir_usd_est[date] = float(value)
        
    print("usd libor. input in format 'YYYY-MM-DD' value")
    date = start_date
    while date < end_date:
        date, value = input().split(sep=' ')
        date = pd.to_datetime(date)
        libor[date] = float(value) / 100

    # res = [interest_rate, libor, current_spot, final_spot, 
    #         ir_rub_est, ir_usd_est, first_leg_payments_periodity, 
    #         second_leg_payments_periodity, margin_bps]
    
    diff_val = construct_credit_swap(
    volume=volume,
    
    start_swap_date=start_date,
    end_swap_date=end_date,
    
    interest_rate=interest_rate, # in percentage i.e. 15.6%
    libor=libor,
    
    current_spot=current_spot, 
    final_spot=current_spot, 
    
    ir_rub_est=ir_rub_est,
    ir_usd_est=ir_usd_est,
    
    first_leg_payments_periodity=first_leg_payments_periodity,
    second_leg_payments_periodity=second_leg_payments_periodity,
    
    margin_bps=margin_bps,
    render=True
)
    
    if make_sensitivity_mx:
        make_sensitivity_matrix(
            final_spots=current_spot,
            libor_val=np.fromiter((val for key, val in libor.items()), dtype=float).mean(),
            percentage_var=10,
            square_grid=5,
            interest_rate=interest_rate,
            
            
        )

    return diff_val
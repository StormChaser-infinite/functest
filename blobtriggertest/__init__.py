import logging
import pandas as pd
import numpy as np
import azure.functions as func
from azure.storage.blob import ContainerClient
import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import math

def clean_df(df):
    """clean the columns of original dataframe into correct data types"""
    date_cols = ['EventStartDate','EventEndDate','CTED','BalloonCTED','CapitalisedPeriodStartDate','StructuredPayment1Date','StructuredPayment2Date',
                 'NextRepaymentDate','InterestRateEffectDateAlteration1','InterestRateEffectDateAlteration2','InterestRateEffectDateAlteration3']
    
    for i in date_cols:
        df[i] = pd.to_datetime(df[i], errors= 'coerce',format= '%Y-%m-%d')
    
    df = df.reset_index()
    df = df.rename(columns= {'index': 'EventID', 'EventID': 'EventID_Orig'})

    return df 



def populate_periods(input):
    """Populate the periods to CTED based on the repayment frequency"""
    input["CTED_select"] = input.apply(lambda x: x['BalloonCTED'] if x['BalloonFlag'] == 'Y' else x['CTED'], axis = 1)
    input = input.set_index('EventID')
    input_data = input[['EventStartDate', 'NextRepaymentDate','RepaymentFrequency', 'CTED_select']].to_dict('index')
    frequency = {'weekly':7, 'fortnightly':14, 'monthly': 1, 'quarterly':3}

    
    df_periods =pd.DataFrame()
    for s in input_data.keys():
        periods = []
        periods.append(input_data[s]['EventStartDate'])
        date_start = input_data[s]['NextRepaymentDate'] 
        date_end =  input_data[s]['CTED_select']
        if input_data[s]['RepaymentFrequency'].lower() in ['monthly', 'quarterly']:
            while date_start < date_end:
                periods.append(date_start)
                date_start = date_start + relativedelta(months = frequency[input_data[s]['RepaymentFrequency'].lower()])
        elif input_data[s]['RepaymentFrequency'].lower() in ['weekly', 'fortnightly']:
            while date_start < date_end:
                periods.append(date_start)
                date_start = date_start + relativedelta(days = frequency[input_data[s]['RepaymentFrequency'].lower()])
        else:
            print(f'The input frequency is not correct for {s}')

        periodid = [ i for i in range(1,len(periods)+1)]
        df_period = pd.DataFrame(list(zip(periodid,periods)), columns= [ 'periodid', 'fromdate'])
        df_period['EventID'] = s
        df_period['todate'] = df_period['fromdate'].shift(-1) + timedelta(days=-1)
        df_period = df_period[df_period['todate'].notnull()]
        df_periods = pd.concat([df_periods,df_period],ignore_index= True)
      
    return df_periods


def calc_avge(date0, date1,date2,date3, rate0, rate1, rate2,rate3, nextpaydate):
    """calculate the weighted average annual inteste rate """
    # periodenddate = dt.datetime.strptime(nextpaydate,'%Y-%m-%d') + timedelta(days=-1)
    dates = [date0, date1, date2,date3,nextpaydate]
    rates = [rate0, rate1, rate2,rate3]
    totaldays = dates[-1] - dates[0]
    # remove the None data
    none_value = []
    for i in range(0,len(dates)):
        if dates[i] == None or pd.isnull(dates[i]):
            none_value.append(i)
    dates_update = [i for j, i in enumerate(dates) if j not in none_value]
    rates_update = [i for j, i in enumerate(rates) if j not in none_value]
    arr = np.array(dates_update)
    noofdays = [i.days for i in np.diff(arr)]
 
    avge_rate = sum([a * b for a, b in zip(rates_update, noofdays)])/(totaldays.days)
    latest_rate = rates_update[-1]

    return [avge_rate,latest_rate]


def calc_structpayment(ids, structpay, todates):
    """calculate the structured payment if the related falg equals to yes"""
    if structpay[ids]['StructuredPaymentsFlag'] == 'Y':
        if structpay[ids]['StructuredPayment1Date'] is not None and structpay[ids]['StructuredPayment1Date'] == (todates + timedelta(days = 1)):
            return structpay[ids]['StructuredPayment1Amt']
        elif structpay[ids]['StructuredPayment2Date'] is not None and  structpay[ids]['StructuredPayment2Date'] == (todates + timedelta(days = 1)):
            return structpay[ids]['StructuredPayment2Amt']
        elif structpay[ids]['StructuredPayment1Date'] is not None and structpay[ids]['StructuredPayment2Date'] is not None and structpay[ids]['StructuredPayment2Date'] == (todates + timedelta(days = 1)) == structpay[ids]['StructuredPayment1Date']:
            return structpay[ids]['StructuredPayment2Amt'] + structpay[ids]['StructuredPayment1Amt']
        else:
            return 0

def product_func(df, structrepay):
    """calculation of p factor, x factor and sum up of p compound"""
    df_factors = pd.DataFrame()
    for i in df.EventID.unique():
        df_sample= df[df['EventID'] == i].reset_index().drop(columns = 'index')
        df_sample['pfactor'] = df_sample.apply(lambda x: 1 if x.periodid == 1 else np.prod(np.array(list(df[(df['periodid'] >= x.periodid) & (df['EventID'] == i)]['compoundperiod']))), axis = 1)
        df_sample['xfactor'] = df_sample.apply(lambda x: np.prod(np.array(list(df[(df['periodid'] >= x.periodid) & (df['EventID'] == i)]['compoundperiod']))), axis = 1)
        df_sample['sumPcompound'] = df_sample['periodid'].apply(lambda x: np.sum(np.array(list(df_sample[(df_sample['periodid'] >= (x + 1) ) & (df_sample['EventID'] == i)]['pfactor'])))+1)
        df_sample['structuredpayment'] = df_sample['todate'].apply(lambda x:  calc_structpayment(i, structrepay, x))
        df_factors = pd.concat([df_factors, df_sample], ignore_index= True)

    return df_factors

def calc_mrc_total(df_factor,inputdata):
    """calculate the total MRC and reducint MRC, the two variables will be ceiling two decimal places"""
    df_mrc_total = df_factor.groupby('EventID', as_index= False).agg({'structuredpayment': 'sum',
                                                                    'fvspmts':'sum',
                                                                    'periodid': 'count',
                                                                    'xfactor':'first',
                                                                    'sumPcompound':'first'})
    df_input = pd.merge(inputdata[['EventID', 'LoanCurrentAmtOwed']].reset_index().drop(columns = 'index'), df_mrc_total, on = 'EventID', how = 'left')
    df_input['mrc_total'] = df_input.apply(lambda x: math.ceil((x.xfactor*x.LoanCurrentAmtOwed-x.fvspmts)/x.sumPcompound *100)/100, axis = 1)
    df_input['mrc_reducing'] = df_input.apply(lambda x: math.ceil((x.LoanCurrentAmtOwed-x.structuredpayment)/x.periodid *100)/100, axis = 1)

    return df_input


def calc_repayment(df_factor, mrc_amount):
    """calculate the repayments and closing balance"""
    df_repayment = pd.DataFrame()
    for i in df_factor['EventID'].unique():
        df_event = df_factor[df_factor.EventID == i].reset_index().drop(columns = 'index')
        for j in range(0, len(df_event)):
            if j == 0:
                interest_payment = [mrc_amount[i]['LoanCurrentAmtOwed'] * df_event.iloc[j]['period_rate'] ]
                reducing_repayment = [interest_payment[j] + mrc_amount[i]['mrc_reducing']]
                repayment =[ [df_event.iloc[j]['table_repayment'] if df_event.iloc[j]['LoanRepaymentType'] == 'Table' else  reducing_repayment][0] + df_event.iloc[j]['structuredpayment']]
                prin_repayment = [repayment[j] - interest_payment[j]]
                closig_bal = [mrc_amount[i]['LoanCurrentAmtOwed'] - prin_repayment[j]]
                index_col = [j+1]
            else:
                interest_payment.append(closig_bal[j-1] * df_event.iloc[j]['period_rate'])
                reducing_repayment.append(interest_payment[j] + mrc_amount[i]['mrc_reducing'])
                if j+1 == len(df_event) or (df_event.iloc[j-1]['structuredpayment'] == 0 and closig_bal[j-1] <= prin_repayment[j-1]):
                    repayment.append(closig_bal[j-1] + interest_payment[j])
                    prin_repayment.append(closig_bal[j-1])
                else:
                    repayment.append([df_event.iloc[j]['table_repayment'] if df_event.iloc[j]['LoanRepaymentType'] == 'Table' else  reducing_repayment][0] + df_event.iloc[j]['structuredpayment'])
                    prin_repayment.append(repayment[j] - interest_payment[j])
                closig_bal.append(closig_bal[j-1] - prin_repayment[j])
                index_col.append(j+1)

        df_repay = pd.DataFrame(list(zip(index_col, interest_payment,reducing_repayment,repayment,prin_repayment ,closig_bal)), 
                                columns= ['periodid', 'interest_payment','reducing_repayment','repayment','prin_repayment','closig_bal'])
        df_repay['EventID'] = i
        df_repayment = pd.concat([df_repayment, df_repay],ignore_index= True)
     
    return df_repayment

def output_final_results(df_comp_repay, start_end_date):
    """prepare the columns for the final output"""
    df_compound_re = pd.merge(df_comp_repay, start_end_date, how = 'left', on = 'EventID')
    df_compound_re['between_start_end_date'] = df_compound_re.apply(lambda x: 1 if x.fromdate >= x.EventStartDate 
                                                                    and x.fromdate < x.EventEndDate else 0, axis= 1 )
    df_pre = df_compound_re.sort_values(by = ['EventID', 'periodid']).groupby(['EventID', 'EventID_Orig'], as_index= False).agg(balance_on_CTED =('closig_bal', 'last'),
                                                                                                                                total_interest_payment =('interest_payment', 'sum'),
                                                                                                                                total_principal_payment =('prin_repayment', 'sum'),
                                                                                                                                table_payment =('table_repayment', 'first'),
                                                                                                                                reducing_payment =('reducing_repayment', 'first'),
                                                                                                                                next_interest_payment =('interest_payment', 'first'),
                                                                                                                                next_principle_payment =('prin_repayment', 'first'),
                                                                                                                                final_repayment_amount = ('repayment', 'last'),
                                                                                                                                final_interest_amount =('interest_payment', 'last'),
                                                                                                                                final_principle_amount =('prin_repayment', 'last'),
                                                                                                                                LoanRepaymentType = ('LoanRepaymentType', 'first'))
    
    df_pre['repayment_calculation'] = df_pre.apply(lambda x: x.table_payment if x.LoanRepaymentType == 'Table' 
                                                         else x.reducing_payment - x.next_interest_payment, axis = 1)
    df_pre['final_fixed_amount_adj'] = df_pre['final_principle_amount'] - df_pre['repayment_calculation']
    df_select = df_compound_re[df_compound_re['between_start_end_date'] == 1].groupby('EventID', as_index= False).agg(total_interest_payment_end_date =('interest_payment', 'sum'),
                                                                                                 total_principal_payment_end_date =('prin_repayment', 'sum'))
    df_final = pd.merge(df_pre,df_select, on = 'EventID', how='left')
    df_final_ouput = df_final[['EventID_Orig', 'balance_on_CTED', 'total_interest_payment','total_principal_payment',
                               'repayment_calculation','next_interest_payment','next_principle_payment','total_interest_payment_end_date','total_principal_payment_end_date',
                               'final_repayment_amount','final_interest_amount','final_principle_amount', 'final_fixed_amount_adj']]
    
    return df_final_ouput




def clac_rates(df_period, inputdata):
    """calculate the weighted average annual inteste rate and period rate in each pay period;
    The rate can only be altered at the first pay period according to the bussiness rule - TP
    Hard code the numbers of days in one year to be 365 - MB"""
    inputdata['annula_interest'] = inputdata.apply(lambda x: calc_avge(x.EventStartDate, x.InterestRateEffectDateAlteration1,x.InterestRateEffectDateAlteration2, x.InterestRateEffectDateAlteration3,
                                                                   x.InterestRateInitital, x.InterestRateAlteration1,x.InterestRateAlteration2,x.InterestRateAlteration3, x.NextRepaymentDate), axis = 1)
    spmet = inputdata.set_index('EventID')[['StructuredPaymentsFlag','StructuredPayment1Date','StructuredPayment1Amt', 
                   'StructuredPayment2Date', 'StructuredPayment2Amt']].to_dict('index')
    df = pd.merge(df_period, inputdata[['EventID', 'LoanRepaymentType', 'annula_interest' ]].reset_index().drop(columns = 'index'), on = 'EventID', how = 'left')
    df['avge_annual_rate'] = df.apply(lambda x: x.annula_interest[0] if x.periodid == 1 else x.annula_interest[1] , axis =1  )
    # df['days'] = df['fromdate'].apply(lambda x: 366 if x.year % 4 == 0 else 365 )
    df['perioddays'] = df.apply(lambda x: (x.todate -  x.fromdate).days + 1 , axis =1)
    df['daily_rate'] =  df.apply(lambda x: x.avge_annual_rate/365, axis =1)
    df['period_rate'] =  df.apply(lambda x: x.daily_rate * x.perioddays, axis =1)
    df['compoundperiod'] =  df['period_rate'] + 1
    df_factor = product_func(df, spmet)
    df_factor = df_factor.fillna(0)
    df_factor['pfactor_shift'] = df_factor['pfactor'].shift(-1).fillna(0)
    df_factor['fvspmts'] = df_factor.apply(lambda x: x.pfactor_shift * x.structuredpayment, axis = 1)
    
    df_mrc_output = calc_mrc_total(df_factor, inputdata)
    mrc_amount = df_mrc_output.set_index('EventID')[['mrc_total', 'mrc_reducing', 'LoanCurrentAmtOwed', 'fvspmts']].to_dict('index')
    
    df_factor['table_repayment'] = df_factor['EventID'].apply(lambda x: mrc_amount[x]['mrc_total'])
    df_repayments = calc_repayment(df_factor, mrc_amount)
    df_factor_repayment = pd.merge(df_factor, df_repayments, on = ['EventID', 'periodid'], how = 'left')
    
    df_final_output = output_final_results(df_factor_repayment, inputdata[['EventID','EventID_Orig' ,'EventStartDate', 'EventEndDate']])

    return df_factor_repayment, df_mrc_output, df_final_output



def export_ouput(df_output):
    '''export the results into the outputs container'''
    output = df_output.to_csv()
    blobService = ContainerClient(account_url = "https://funcdemo.blob.core.windows.net", 
                                   credential= "e4MRGQUsGoQLwqIQw2pw5fEVSonqVoSpMJV1X0QSZ6gaYmXTaE6aLdz4n6a8BD18wmRa/qbSsU5I+AStB1JRKg==",
                                   container_name = "outputs")
    file_name = 'mrc_calculation' + dt.date.today().strftime("%Y%m%d") + '.csv'
    blobService.upload_blob(file_name, output, overwrite=True, encoding='utf-8')


def main(myblob: func.InputStream):
    '''calculate the mrc'''
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n")
    
    df_input = pd.read_csv(myblob)
    df_input_cleaned = clean_df(df_input)
    df_populate_period = populate_periods(df_input_cleaned)
    factors, mrc, outputs = clac_rates(df_populate_period, df_input_cleaned)

    export_ouput(outputs)


    

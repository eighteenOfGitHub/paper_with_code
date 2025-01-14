import os
import openai

openai.api_key = ''
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
import pandas as pd
from data1.small_context import get_datasets
import re
# from llama_utils import llama_api_qa
from data1.serialize import SerializerSettings

# Add situation description by each dataset

model_select = 'gpt-4-1106-preview'
gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=0.5,
    top_p=0.5,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

hyper_gpt35 = f"Set hyperparameters to: temp={gpt3_hypers['temp']}, " \
                           f"alpha={gpt3_hypers['alpha']}, beta={gpt3_hypers['beta']}, " \
                           f"basic={gpt3_hypers['basic']}, settings={gpt3_hypers['settings']}"

hyper_gpt4 = f"Set hyperparameters to: alpha={gpt4_hypers['alpha']}, " \
                           f"temp={gpt4_hypers['temp']}, top_p={gpt4_hypers['top_p']}, " \
                           f"basic={gpt4_hypers['basic']}, settings={gpt4_hypers['settings']}"
def paraphrase_initial(data_name):
    if data_name == 'AirPassengersDataset':
        desp = "This is a series of monthly passenger numbers for international flights, " \
               "where each value is in thousands of passengers for that month. "
    if data_name == 'AusBeerDataset':
        desp = "This is a quarterly series of beer production, where each value is " \
               "the kilolitres of beer produced in that quarter. "
    if data_name == 'GasRateCO2Dataset':
        desp = "This is a time series dataset describing monthly carbon dioxide emissions. "
    if data_name == 'MonthlyMilkDataset':
        desp = "This is a time-series data set describing monthly milk production, " \
               "Each number is the average number of tons of milk produced by each cow during the month. "
    if data_name == 'SunspotsDataset':
        desp = "This is a dataset that records the number of sunspots in each month, " \
               "where each data is the number of sunspots in that month. "
    if data_name == 'WineDataset':
        desp = "This is a dataset of monthly wine production in Australia," \
               "where each figure is the number of wine bottles produced in that month. "
    if data_name == 'WoolyDataset':
        desp = "This is an Australian yarn production for each quarter, " \
               "where each value is how many tons of yarn were produced in that quarter. "
    if data_name == 'HeartRateDataset':
        desp = "The series contains 1800 uniformly spaced instantaneous " \
               "heart rate measurements from a single subject. "

    return desp


# Transfer Sequences to Natural Language.
# seq: <class 'pandas.core.series.Series'>, des: String
def paraphrase_seq2lan(seq, desp):
    results = ''
    # The values of the sequence are read one by one and a description is output
    for i in range(len(seq) - 1):
        t1 = seq.iloc[i]  # Select elements by position
        t2 = seq.iloc[i + 1]  # Select next elements by position
        result = describe_change(t1, t2)
        results += result
    lan = desp + results

    return lan


def describe_change(t1, t2):
    if t2 > t1:
        return f"from {t1} increasing to {t2}, "
    elif t2 < t1:
        return f"from {t1} decreasing to {t2}, "
    else:
        return f"it remains flat from {t2} to {t1}，"


# Recover from language description to sequence
def recover_lan2seq(input_string):
    # step 1: cut description
    dot_index = input_string.find('.')
    cleaned_string = input_string[dot_index + 1:].strip() if dot_index != -1 else input_string.strip()

    # Step 2: task numbers
    numbers = re.findall(r'(\d+\.\d+)', cleaned_string)
    # Transfer to list
    float_numbers = [float(num) for num in numbers]

    # Step 3: Kill the doubled numbers
    filtered_numbers = [float_numbers[i] for i in range(len(float_numbers)) if i % 2 == 0]
    # add the last one
    filtered_numbers.append(float_numbers[-1])
    # recover to pandas Series
    result_series = pd.Series(filtered_numbers)

    return result_series


def recover_lan2seq_llm(input_string):
    # Step 2: task numbers
    numbers = re.findall(r'(\d+\.\d+)', input_string)
    # Transfer to list
    float_numbers = [float(num) for num in numbers]

    # Step 3: Kill the doubled numbers
    filtered_numbers = [float_numbers[i] for i in range(len(float_numbers)) if i % 2 == 0]
    # add the last one
    filtered_numbers.append(float_numbers[-1])
    # recover to pandas Series
    result_series = pd.Series(filtered_numbers)

    return result_series


def paraphrase_nlp(datasets_list):
    # Train_lan: the paraphrased train sequence
    # Test_lan: the paraphrased test sequence
    # seq_test: pandas sequence of test
    datasets = get_datasets()
    for dataset_name in datasets_list:
        desp = paraphrase_initial(dataset_name)
        data = datasets[dataset_name]
        train, test = data
        print("Train len:", train.shape)
        print("test len:", test.shape)
        Train_lan = paraphrase_seq2lan(train, desp)
        Test_lan = paraphrase_seq2lan(test, desp)
        seq_test = recover_lan2seq(Test_lan)
        print("seq pred len:", seq_test.shape)
        if test.shape != seq_test.shape:
            print("Warning! The data lost!")

    return Train_lan, Test_lan, seq_test


def paraphrase_llm(datasets_list):
    prompt = " analyze this time series and rewrite it" \
             " as a trend-by-trend representation of discrete values. Only numerical " \
             "changes are described, not date changes. For example, the template like {from 1.0 increasing to 2.0, " \
             "from 2.0 decreasing to 0.5,} Be careful not to lose every sequence value. "
    datasets = get_datasets()

    for dataset_name in datasets_list:
        desp = paraphrase_initial(dataset_name)
        data = datasets[dataset_name]
        train, test = data
        content_train = "You are a useful assistant," + desp + str(train)
        content_test = "You are a useful assistant," + desp + str(test)
        response = openai.ChatCompletion.create(
            model=model_select,
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content_train}
            ]
        )
        Train_lan = response.choices[0].message.content
        response = openai.ChatCompletion.create(
            model=model_select,
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content_test}
            ]
        )
        Test_lan = response.choices[0].message.content
        print("Test_lan:", Test_lan)
        seq_test = recover_lan2seq_llm(Test_lan)
        if test.shape != seq_test.shape:
            print("the process error!")
            print("seq_test.shape:", seq_test.shape)
            print("test.shape:", test.shape)

    return Train_lan, Test_lan, seq_test

def paraphrasing_predict_llm(desp, train_lan, steps, model_name):
    if model_name == 'gpt-3.5-turbo-0125':
        hyper_parameters_message = hyper_gpt35
    else:
        hyper_parameters_message = hyper_gpt4
    prompt = "You are a helpful assistant that performs time series predictions. " \
             "The user will provide a sequence and you will predict the remaining sequence." \
             "The sequence is represented by decimal strings separated by commas. " \
             "Please continue the following sequence without producing any additional text. " \
             "Do not say anything like 'the next terms in the sequence are', just return the numbers."
    # prompt_add = f"Please predict ahead in {steps} steps, one step means (from 1.0 increasing to 2.0,) or" \
    #              "(from 2.0 decreasing to 0.5,), The final output follows exactly steps. Sequence:\n"
    prompt_add = (f"Predict the next {steps} steps, where each step follows the format (starting from 1.0 and increasing to 2.0) or (starting from 2.0 and decreasing to 0.5)."\
                  " The final output should precisely follow the specified number of steps. Provide a sequence:\n")

    content_train = prompt + desp + prompt_add + train_lan
    response = openai.ChatCompletion.create(
        model=model_name,
        response_format={"type": "text"},
        messages=[
            {"role": "system", "content": hyper_parameters_message},
            {"role": "system", "content": prompt},
            {"role": "user", "content": content_train}
        ]
    )
    Test_lan = response.choices[0].message.content
    # print("Test_lan:", Test_lan)
    seq_test = recover_lan2seq_llm(Test_lan)

    return seq_test


def paraphrasing_predict_llama(desp, train_lan, steps, model_name):
    # prompt = "You are a helpful assistant that performs time series predictions. " \
    #          "The user will provide a sequence and you will predict the remaining sequence." \
    #          "The sequence is represented by decimal strings separated by commas. " \
    #          "Please continue the following sequence without producing any additional text. " \
    #          "Do not say anything like 'the next terms in the sequence are', just return the numbers."
    prompt_add = f"Please predict ahead in {steps} steps, one step means (from 1.0 increasing to 2.0,) or" \
                 "(from 2.0 decreasing to 0.5,), The final output follows exactly steps. Sequence:\n"
    content_train = desp + prompt_add + train_lan
    response = llama_api_qa(model_name, content_train)
    seq_test = recover_lan2seq_llm(response)

    return seq_test


# test main
if __name__ == '__main__':
    # initial
    datasets_list = [
        'AirPassengersDataset',
        'AusBeerDataset',
        'GasRateCO2Dataset',
        'MonthlyMilkDataset',
        'SunspotsDataset',
        'WineDataset',
        'WoolyDataset',
        'HeartRateDataset',
    ]

    # traditional
    paraphrase_nlp(datasets_list)

    # LLM
    paraphrase_llm(datasets_list)

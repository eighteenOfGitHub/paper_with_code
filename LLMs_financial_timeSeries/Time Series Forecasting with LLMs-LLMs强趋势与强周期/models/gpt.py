from data1.serialize import serialize_arr, SerializerSettings
import openai
import tiktoken
import numpy as np
from jax import grad,vmap
import google.generativeai as genai


safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

def tokenize_fn(str, model):
    """
    Retrieve the token IDs for a string for a specific GPT model.

    Args:
        str (list of str): str to be tokenized.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model(model)
    # tiktoken.get_encoding()
    return encoding.encode(str)

def get_allowed_ids(strs, model):
    """
    Retrieve the token IDs for a given list of strings for a specific GPT model.

    Args:
        strs (list of str): strs to be converted.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model(model)
    ids = []
    for s in strs:
        id = encoding.encode(s)
        ids.extend(id)
    return ids

def gemini_completion_fn(model, input_str, steps, settings, num_samples, temp, whether_blanket=True, genai_key=None):
    """
    Generate text completions from Gemini using Google's API.

    Args:
        model (str): Name of the Gemini model to use.
        input_str (str): Serialized input time series data1.
        steps (int): Number of time steps to predict.
        settings (SerializerSettings): Serialization settings.
        num_samples (int): Number of completions to generate. (表示选择的数量?)
        temp (float): Temperature for sampling.

    Returns:
        list of str: List of generated samples.
    """
    # model = genai.GenerativeModel(model)
    # avg_tokens_per_step = model.count_tokens(input_str)
    # avg_tokens_per_step = len(tokenize_fn(input_str, 'gpt-3.5-turbo')) / len(input_str.split(settings.time_sep))
    # gpt不能直接用内部的token_count, 加上起始符,终止符等等额外的操作,会多一些token; 而gemini是遵从于原本的单词数量, 却对于单词忽略了空格 (可能认为空格也是单词中一部分吧), 对于数字型计入空格

    # 这里得到input str的 token 长度 (gemini可以直接调用得到)
    # define logit bias to prevent GPT-3 from producing unwanted tokens
    if genai_key is not None:
        genai.configure(api_key=genai_key, transport='rest')
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)]
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
    if not whether_blanket:
        input_str = input_str.replace(" ", "")
    if (model not in ['gemini-1.0-pro','gemini-pro']): # logit bias not supported for chat models, 对于gemini此处也如此设置
        logit_bias = {id: 30 for id in get_allowed_ids(allowed_tokens, model)}
    # print(input_str)  # 对于 period_prediction, 如果需要得到原文中 renormalization 后序列, 可以在这里打印, 然后复制到响应文件里input_str里
    #  由于此处是免费版, candidate限制只有一个, 所以此处采用反复调用模型
    # assert False
    if model in ['gemini-1.0-pro', 'gemini-pro']:  # 此处为主要修改位置
        gemini_sys_message = f"You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence for {steps*10} steps. The sequence is represented by decimal strings separated by commas, and each step consists of contents between two commas."
        # gemini_sys_message = f"You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        content_fin = []
        model = genai.GenerativeModel(model)
        # model.count_tokens(input_str)
        for i in range(num_samples):
            print("Index:", i)
            response = model.generate_content(
                contents=gemini_sys_message+extra_input+input_str+settings.time_sep,  # 此处因为只有一个 contents 选项, 所以将所有的话语都放在一起
                generation_config=genai.types.GenerationConfig(
                    temperature=temp,
                    # candidate_count=num_samples,
                    # max_output_tokens=int(avg_tokens_per_step*steps)
                    # 这里是根据input length的平均token长度来进行预测 (可以使得预测结果更有效率,缺点是可能会缺失部分结果_可能会有影响,此处先忽略)
                    # 在其他代码里尝试过, 不知为何原因用不了
                    # 此处对于输出的长度, 直接用max_output_tokens来控制 (这里可以避免预测过长, 浪费钱.....
                    # The free version of the API only offers one candidate option...
                ),
                safety_settings=safety_settings
            )
            tmp = response.text
            if not whether_blanket:  # 为了保证最终输出不会受到影响
                tmp = ' '.join(response.text)
            content_fin.append(tmp)
        # 这里取出了输出,并且将其作为list展示
        return content_fin
    # 注意这里是
    else:
        assert False

def gpt_completion_fn(model, input_str, steps, settings, num_samples, temp, whether_blanket=True):
    """
    Generate text completions from GPT using OpenAI's API.

    Args:
        model (str): Name of the GPT-3 model to use.
        input_str (str): Serialized input time series data1.
        steps (int): Number of time steps to predict.
        settings (SerializerSettings): Serialization settings.
        num_samples (int): Number of completions to generate.
        temp (float): Temperature for sampling.

    Returns:
        list of str: List of generated samples.
    """
    avg_tokens_per_step = len(tokenize_fn(input_str, model)) / len(input_str.split(settings.time_sep))
    # define logit bias to prevent GPT-3 from producing unwanted tokens
    logit_bias = {}
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
    if (model not in ['gpt-3.5-turbo','gpt-4']): # logit bias not supported for chat models
        logit_bias = {id: 30 for id in get_allowed_ids(allowed_tokens, model)}
    if model in ['gpt-3.5-turbo','gpt-4']:
        chatgpt_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                    {"role": "system", "content": chatgpt_sys_message},
                    {"role": "user", "content": extra_input+input_str+settings.time_sep}
                ],
            max_tokens=int(avg_tokens_per_step*steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples,  # 这里默认有10个candidates,此处设置 candidates, 这么多candidates然后怎么选择???
        )
        return [choice.message.content for choice in response.choices]  # 这里返回多个candidate结果 (感觉可以近似认为是跑多次)
    else:
        response = openai.Completion.create(
            model=model,
            prompt=input_str, 
            max_tokens=int(avg_tokens_per_step*steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples
        )
        return [choice.text for choice in response.choices]
    
def gpt_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1):
    """
    Calculate the Negative Log-Likelihood (NLL) per dimension of the target array according to the LLM.

    Args:
        model (str): Name of the LLM model to use.
        input_arr (array-like): Input array (history).
        target_arr (array-like): Ground target array (future).
        settings (SerializerSettings): Serialization settings.
        transform (callable): Transformation applied to the numerical values before serialization.
        count_seps (bool, optional): Whether to account for separators in the calculation. Should be true for models that generate a variable number of digits. Defaults to True.
        temp (float, optional): Temperature for sampling. Defaults to 1.

    Returns:
        float: Calculated NLL per dimension.
    """
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    assert input_str.endswith(settings.time_sep), f'Input string must end with {settings.time_sep}, got {input_str}'
    full_series = input_str + target_str
    response = openai.Completion.create(model=model, prompt=full_series, logprobs=5, max_tokens=0, echo=True, temperature=temp)
    #print(response['choices'][0])
    logprobs = np.array(response['choices'][0].logprobs.token_logprobs, dtype=np.float32)
    tokens = np.array(response['choices'][0].logprobs.tokens)
    top5logprobs = response['choices'][0].logprobs.top_logprobs
    seps = tokens==settings.time_sep
    target_start = np.argmax(np.cumsum(seps)==len(input_arr)) + 1
    logprobs = logprobs[target_start:]
    tokens = tokens[target_start:]
    top5logprobs = top5logprobs[target_start:]
    seps = tokens==settings.time_sep
    assert len(logprobs[seps]) == len(target_arr), f'There should be one separator per target. Got {len(logprobs[seps])} separators and {len(target_arr)} targets.'
    # adjust logprobs by removing extraneous and renormalizing (see appendix of paper)
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign, settings.bit_sep+settings.decimal_point]
    allowed_tokens = {t for t in allowed_tokens if len(t) > 0}
    p_extra = np.array([sum(np.exp(ll) for k,ll in top5logprobs[i].items() if not (k in allowed_tokens)) for i in range(len(top5logprobs))])
    if settings.bit_sep == '':
        p_extra = 0
    adjusted_logprobs = logprobs - np.log(1-p_extra)
    digits_bits = -adjusted_logprobs[~seps].sum()
    seps_bits = -adjusted_logprobs[seps].sum()
    BPD = digits_bits/len(target_arr)
    if count_seps:
        BPD += seps_bits/len(target_arr)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll-avg_logdet_dydx

import Numerical_Analysis2
import agent_curve_to_picture  #
import catchpic   #
import model_diff   #
import opinion_jump   #
import picinfluence_matrix_all
import picsubbornness_matrix_all
import simple_cell_inf
import true_inf
import inf_model_diff
import average_opinion
'''
data_deepseek
data_gpt
data_qwenflash
data_qwen3.5
data_gemini
data_llama
'''
'''
{
agent_curve_to_picture / true_inf /2 matrix
用来说明RQ1：身份不同的智能体是否表现出差异化的传播动力学
}
{
Numerical_Analysis2 / simple_cell_inf
用来说明RQ2：在多智能体环境中，身份线索（性别、种族）是否系统性地影响个体在信息传播中的影响力与固执程度？
}
RQ1 && RQ2 diff: RQ1:pos and neg ;RQ2:inf and stu
{
average_opinion
用来说明RQ3：在多轮互动中，LLM群体的信念演化是否系统性地偏向真实信息或谣言信息？
}
{
average_opinion /2个matrix /model_diff/inf_model_diff
用来说明RQ4：不同认知属性的谣言（如高不确定性、高威胁性）是否表现出差异化的传播动力学？
}
{
average_opinion / 2个matrix /model_diff/inf_model_diff/opinion_jump(special)
用来说明RQ5：不同LLM在信息真实性与社会偏见方面是否表现出一致的行为模式？
}
RQ4和RQ5对称
'''
lis = ["data_deepseek",
"data_gpt","data_qwenflash",
"data_qwen3.5","data_llama",
"data_gemini"]
def run(path):
    # print(path)
    # average_opinion.step(path)
    #
    # agent_curve_to_picture.step(path)
    # model_diff.step()
    picinfluence_matrix_all.main()
    picsubbornness_matrix_all.main()
    #
    # inf_model_diff.step(path)
    #
    # agent_curve_to_picture.step(path)

## generate reports
    # true_inf.step(path)
    # simple_cell_inf.step(path)
    # Numerical_Analysis2.step(path)
    # opinion_jump.step(path)
    #
    # catchpic.step(path)
# for i in lis:
for i in ["1"]:
    run(i)

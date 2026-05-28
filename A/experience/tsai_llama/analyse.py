# -*- coding: utf-8 -*-
import json
import matplotlib
matplotlib.use('Agg')#线程问题
import matplotlib.pyplot as plt
import os

def step(log_dir):
    event_file = os.path.join(log_dir, "event.txt")
    if not os.path.exists(event_file):
        print(f"the file is not exist: {event_file}")
        return

    # 从目录名解析支持者ID和反对者ID
    dir_name = os.path.basename(log_dir)
    parts = dir_name.split('_')
    support_part = None
    oppose_part = None
    for part in parts:
        if part.startswith('support'):
            support_part = part
        elif part.startswith('oppose'):
            oppose_part = part
    if support_part is None or oppose_part is None:
        print(f"Unable to resolve from directory name support/oppose: {dir_name}")
        return
    idsup = int(support_part.replace('support', ''))
    idop = int(oppose_part.replace('oppose', ''))
    stubborn = not (parts[-1] == "inf")  # whether it is influenced

    # read all reflect logs
    data = []  # item format: (ts, agent_id, opinion)
    with open(event_file,encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get('type') == 'reflect':
                    ts = record['ts']
                    owner = record['owner']
                    agent_id = int(owner.split('_')[1])
                    opinion = record['item']['new_opinion']
                    data.append((ts, agent_id, opinion))
            except Exception as e:
                print(f"error: {e}")

    if not data:
        print("no reflect data")
        return

    data.sort(key=lambda x: x[0])
    # 按智能体分组
    agent_opinions = {}
    for ts, aid, op in data:
        if aid not in agent_opinions:
            agent_opinions[aid] = {'ts': [], 'opinions': []}
        agent_opinions[aid]['ts'].append(ts)
        agent_opinions[aid]['opinions'].append(op)

    # 绘图
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(range(6))
    offsets = [-0.02, -0.01, 0.01, 0.02]  # 偏移数组(图更清晰)
    offset_index = 0
    for i, (aid, vals) in enumerate(agent_opinions.items()):
        ts_vals = vals['ts']
        op_vals = vals['opinions']
        if aid == idsup:
            label = f'Agent {aid} (rumor)'
            linewidth = 3
            marker = 'o'
            offset = 0
        elif aid == idop:
            label = f'Agent {aid} (truth)'
            linewidth = 3
            marker = 's'
            offset = 0
        else:
            label = f'Agent {aid}'
            linewidth = 1
            marker = '.'
            offset = offsets[offset_index % len(offsets)]
            offset_index += 1
        op_vals_offset = [v + offset for v in op_vals]
        plt.plot(ts_vals, op_vals_offset, label=label, color=colors[i % len(colors)],
                 linewidth=linewidth, marker=marker)

    plt.xlabel('Time Step')
    plt.ylabel('Opinion')
    title = f'Opinion Evolution in {dir_name}'
    if stubborn:
        title += ' (Stubborn)'
    plt.title(title)
    # 图例放置在主图外部右侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.grid(True)
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1])

    output_img = os.path.join(log_dir, 'opinion_trend.png')
    plt.savefig(output_img)
    plt.close()
    print(f"save the image to {output_img}")

def main():
    # 测试用
    step('./logs/topic0_support0_oppose3_inf')

if __name__ == "__main__":
    main()
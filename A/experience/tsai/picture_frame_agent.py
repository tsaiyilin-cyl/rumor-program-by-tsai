# -*- coding: utf-8 -*-
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 设置中文字体及全局字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 14})   # 全局字体放大一倍

def step(log_dir):
    event_file = os.path.join(log_dir, "event.txt")
    if not os.path.exists(event_file):
        print(f"the file is not exist: {event_file}")
        return

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
    stubborn = not (parts[-1] == "inf")

    data = []
    with open(event_file, encoding="utf-8") as f:
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
    agent_opinions = {}
    for ts, aid, op in data:
        if aid not in agent_opinions:
            agent_opinions[aid] = {'ts': [], 'opinions': []}
        agent_opinions[aid]['ts'].append(ts)
        agent_opinions[aid]['opinions'].append(op)

    plt.figure(figsize=(10, 6))
    offsets = [-0.03, -0.015, 0.015, 0.03]
    offset_index = 0

    for aid, vals in agent_opinions.items():
        ts_vals = vals['ts']
        op_vals = vals['opinions']

        if aid == idsup:
            color = 'red'
            label = '谣言方'
            linewidth = 3
            marker = 'o'
            offset = 0
        elif aid == idop:
            color = 'green'
            label = '真相方'
            linewidth = 3
            marker = 's'
            offset = 0
        else:
            color = 'dimgray'
            label = ''
            linewidth = 1
            marker = '.'
            offset = offsets[offset_index % len(offsets)]
            offset_index += 1

        op_vals_offset = [v + offset for v in op_vals]
        plt.plot(ts_vals, op_vals_offset, label=label, color=color,
                 linewidth=linewidth, marker=marker)

    from matplotlib.lines import Line2D
    neutral_handle = Line2D([], [], color='dimgray', linewidth=1, marker='.', label='中立方')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(neutral_handle)
    labels.append('中立方')

    plt.xlabel('辩论轮数')
    plt.ylabel('观点值')
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., prop={'size': 14})
    plt.subplots_adjust(right=0.75)

    plt.grid(True)
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1])

    # 设置横坐标步长为1，范围0~10
    plt.xlim(0, 10)
    plt.xticks(range(0, 11, 1))

    output_img = os.path.join(log_dir, 'opinion_trend.png')
    plt.savefig(output_img)
    plt.close()
    print(f"save the image to {output_img}")

def main():
    step('./logs/topic3_support2_oppose1_inf')
    step('./logs/topic4_support5_oppose0_inf')

if __name__ == "__main__":
    main()
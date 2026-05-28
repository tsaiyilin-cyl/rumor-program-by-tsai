import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

plots_dir = Path(__file__).parent / "nodirect graph"
plots_dir.mkdir(exist_ok=True)

G = nx.complete_graph(6)
pos = nx.circular_layout(G)

plt.figure(figsize=(6, 6))
colors = ['lightblue'] * 6
colors[1] = 'red'
colors[5] = 'green'
nx.draw_networkx_nodes(
    G, pos,
    node_color="none",
    edgecolors=colors,
    linewidths=2.5,
    node_size=800
)
nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)

legend_handles = [
    mpatches.Patch(facecolor='none', edgecolor='red',   linewidth=2, label='rumor'),
    mpatches.Patch(facecolor='none', edgecolor='green', linewidth=2, label='truth'),
    mpatches.Patch(facecolor='none', edgecolor='lightblue',  linewidth=2, label='neutral')
]
plt.legend(handles=legend_handles, loc='upper left', fontsize=10)

# 去掉坐标轴，添加标题
plt.axis('off')
plt.title("Community Network", fontsize=14)
save_path = plots_dir / f'nondirect graph.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
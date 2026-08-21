import os
import sys
import matplotlib.pyplot as plt

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_path)
from gate_csv_parser import load_gates_2D

current_path = os.path.dirname(os.path.abspath(__file__))
filename = "ref_gates_autoX.csv"

file_path = os.path.join(parent_path, filename)
gates = load_gates_2D(file_path)

# gates columns: [gate_index, x1, y1, x2, y2]
fig, ax = plt.subplots()

for gate in gates:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    ax.plot([x1, x2], [y1, y2], 'b-')
    bot_x = x1 if y1 < y2 else x2
    bot_y = min(y1, y2)
    ax.annotate(int(gate[0]), xy=(bot_x, bot_y), ha='center', va='top',
                xytext=(0, -4), textcoords='offset points')

start = gates[0]
end = gates[-1]

for gate, label in [(start, 'Start'), (end, 'End')]:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    top_x = x1 if y1 > y2 else x2
    top_y = max(y1, y2)
    ax.text(top_x, top_y + 1, label, ha='center')

ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
plt.show()





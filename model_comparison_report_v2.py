"""
model_comparison_report_v2.py
-------------------------------
Reads model_comparison_v2.csv (written by train_multimodel_v2.py) and produces
a markdown comparison table and an accuracy-vs-size bar/line chart.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('model_comparison_v2.csv')
df_sorted = df.sort_values('val_RMSE_C')

with open('model_comparison_v2.md', 'w') as f:
    f.write("# Model Comparison (v2)\n\n")
    f.write(df_sorted.to_markdown(index=False))
    f.write("\n\n_Latency (per-model inference time in microseconds) is measured on-device "
            "by the firmware and logged separately -- see the Live_*_us columns in the "
            "firmware's serial CSV output. Update this table manually with those values "
            "once firmware testing is complete._\n")

labels = df_sorted['model_name'] + '_' + df_sorted['precision']

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(labels, df_sorted['val_RMSE_C'], color='steelblue')
ax1.set_ylabel('Validation RMSE (deg C)')
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax2 = ax1.twinx()
ax2.plot(labels, df_sorted['size_bytes'], color='darkorange', marker='o')
ax2.set_ylabel('Model size (bytes)')
plt.title('Accuracy vs Model Size Trade-off (v2)')
plt.tight_layout()
plt.savefig('model_comparison_v2.png', dpi=150)
print("Wrote model_comparison_v2.md and model_comparison_v2.png")

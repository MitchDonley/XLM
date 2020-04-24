import glob
import pdb
import torch
import seaborn as sn
from matplotlib import pyplot as plt

XNLI_langs = ["Arabic", "Bulgarian", "German", 'Greek', "English", 'Spanish', 'French', 'Hindi', 'Russian', 'Swahili', 'Thai', 'Turkish', 'Urdu', 'Vietnamese', "Chinese"]
XNLI_labels = ['contradiction','neutral','entailment']

# model and epoch to load from
# model = 'test_xnli_mlm_tlm/8av043orrl/' # THIS MODEL HAS INVALID CONFUSION MATRICES!!
model = 'test_xnli_tlm_fine_tune/' # THIS MODEL HAS CORRECT CONFUSION MATRICES!!

# load confusion matrices
mat_files = glob.glob('./dumped/' + model + 'conf_mats_epoch*')
conf_mats = torch.zeros((len(mat_files), 15, 3, 3)) # size is specific to XNLI, which is 15 languages with a 3-way classification
for i, file in enumerate(mat_files):
    mat = torch.load(file)
    conf_mats[i] = mat

# plot test accuracy over epochs (average and per language)
accu_per_lang = torch.zeros((conf_mats.shape[0], 15))
accu_total = torch.zeros(conf_mats.shape[0])
for i in range(conf_mats.shape[0]):
    for j in range(15):
        accu_per_lang[i,j] = conf_mats[i,j].diag().sum() / conf_mats[i,j].sum()
    accu_total[i] = accu_per_lang[i].sum() / 15

plt.figure()
ax = plt.subplot(111)
for i, epoch in enumerate(accu_per_lang.T):
    ax.plot(epoch, alpha=0.7, label=XNLI_langs[i])
ax.plot(accu_total, color='blue', linewidth=3, label='Average')

ax.set_title("XNLI Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.grid(True)

# put legend outide of plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('./dumped/' + model + 'accu_plot.png', dpi=300)


# plot confusion matrices from a single epoch

# choose the epoch with the highest average accuracy
epoch = torch.argmax(accu_total).item()

plt.figure()
fig, axs = plt.subplots(3, 5)
cbar_ax = fig.add_axes([.91, .3, .015, .4])
cnt = 0
for j in range(5):
    for i in range(3):
        axs[i,j].set_title(XNLI_langs[cnt])
        sn.heatmap((conf_mats[epoch, cnt] / conf_mats[epoch, cnt].sum()), ax=axs[i,j], cbar= cnt == 0, cbar_ax=None if i else cbar_ax)
        cnt += 1
fig.tight_layout(rect=[0, 0, .9, 1])

plt.savefig('./dumped/' + model + 'conf_mat_langs.png', dpi=300)

# confusion matrix for all languages
plt.figure()
# hm = sn.heatmap((conf_mats[epoch].sum(dim=0) / conf_mats[epoch].sum(dim=0).sum()), annot=True, fmt='.3g', xticklabels=XNLI_labels, yticklabels=XNLI_labels)
hm = sn.heatmap((conf_mats[epoch].sum(dim=0) / conf_mats[epoch].sum(dim=0).sum()), annot=True, fmt='.3g')
hm.set_xticklabels(XNLI_labels, fontsize=7)
hm.set_yticklabels(XNLI_labels, fontsize=7)
plt.title('All Languages')
plt.ylabel('Ground Truth')
plt.xlabel('Predicted')
plt.savefig('./dumped/' + model + 'conf_mat_total.png', dpi=300)

print('hi')
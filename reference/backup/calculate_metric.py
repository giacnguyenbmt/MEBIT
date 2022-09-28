import os
import glob
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

if __name__ == '__main__':
	gt_dir = sys.argv[1]
	dt_dir = sys.argv[2]

	gt = []
	dt = []
	gt_file_list = glob.glob(os.path.join(gt_dir, '*.txt'))

	for i, gt_file_path in enumerate(gt_file_list):
		gt_file_name = os.path.split(gt_file_path)[-1]
		dt_file_path = os.path.join(dt_dir, gt_file_name)

		with open(gt_file_path) as f:
			gt.append(f.read())

		with open(dt_file_path) as f:
			dt.append(f.read())

	acc = accuracy_score(gt, dt)
	p = precision_score(gt, dt, average='weighted')
	r = recall_score(gt, dt, average='weighted')
	f1 = f1_score(gt, dt, average='weighted')
	cm = confusion_matrix(gt, dt, normalize=None)

	fig, ax = plt.subplots(figsize=(8,6), dpi=100)
	sns.heatmap(cm, cmap='Blues', square = True, annot=True, fmt='g', ax=ax)
	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
	ax.set_title('Confusion Matrix')
	ax.yaxis.set_ticklabels(['Green', 'None', 'Red', 'Yellow'])
	ax.xaxis.set_ticklabels(['Green', 'None', 'Red', 'Yellow'])
	plt.show()

	print('acc =', acc)
	print('f1 =', f1)
	print('precision =', p)
	print('recall =', r)
	print(cm)
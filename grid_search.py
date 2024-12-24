'''
Created on Jan. 22, 2024

@author: Yihao Fang
'''
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x=[0.3,0.3,0.3,0.3,0.3,0.25,0.25,0.25,0.25,0.2]
y=[0.6,0.55,0.5,0.45,0.4,0.6,0.55,0.5,0.45,0.6]
z=[0.1,0.15,0.2,0.25,0.3,0.15,0.2,0.25,0.3,0.2]
c=[0.251572327,0.264150943,0.27672956,0.245283019,0.238993711,0.257861635,0.27672956,0.251572327,0.257861635,0.245283019]

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#dataset = 'open-squad'
#dataset = 'hotpotqa'
dataset = 'fever'
sent_len = 'medium'

df = pd.read_csv("log/%s-%s_inst_by_inst.csv"%(dataset,sent_len))
df = df.query("Question == 'EM' or Question == 'F1'")
df.set_index('Question', inplace=True)
df.drop("GT Answer", axis=1, inplace=True)
df = df.transpose()

df = df.reset_index()
df = df.rename(columns={'index': 'Experiment'})

df['Alpha'] = df.apply(lambda x: eval(x['Experiment'].split('_')[1].split('+')[-2])[0], axis=1)
df['Beta'] = df.apply(lambda x: eval(x['Experiment'].split('_')[1].split('+')[-2])[1], axis=1)
df['Gamma'] = df.apply(lambda x: eval(x['Experiment'].split('_')[1].split('+')[-2])[2], axis=1)
df['W_1'] = df.apply(lambda x: eval(x['Experiment'].split('_')[1].split('+')[-1])[0], axis=1)
df['W_2'] = df.apply(lambda x: eval(x['Experiment'].split('_')[1].split('+')[-1])[1], axis=1)
df['W_3'] = df.apply(lambda x: eval(x['Experiment'].split('_')[1].split('+')[-1])[2], axis=1)

df.drop("Experiment", axis=1, inplace=True)

df['EM'] = df.apply(lambda x: float(x['EM']), axis=1)
if dataset != 'fever':
    df['F1'] = df.apply(lambda x: float(x['F1']), axis=1)

df = df.query("W_1 == 0.15 and W_2 == 0.55 and W_3 == 0.3 or W_1 == 0.2 and W_2 == 0.55 and W_3 == 0.25 or W_1 == 0.3 and W_2 == 0.6 and W_3 == 0.1 or W_1 == 0.3 and W_2 == 0.5 and W_3 == 0.2 or W_1 == 1.0 and W_2 == 0.0 and W_3 == 0.0")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
   
   
x_labels = {'Alpha':r'$\alpha$', 'Beta':r'$\beta$', 'Gamma':r'$\gamma$', 'W_1':r'$w_1$', 'W_2':r'$w_2$', 'W_3':r'$w_3$'}
 
'''    
fig, axes = plt.subplots(nrows=1,ncols=6,figsize=(12,2))

for ax, x in zip(axes, ['Alpha', 'Beta', 'Gamma', 'W_1', 'W_2', 'W_3']):
    ax.margins(x=0.01)
    g=sns.lineplot(data=df, x=x, y="EM", ax=ax)
    ylabels = ['{:,.2f}'.format(x) + '' for x in g.get_yticks()]
    g.set_yticklabels(ylabels)
    ax.set_xlabel(x_labels[x])

fig.tight_layout() 
#fig.subplots_adjust(top=0.7)
'''

g = sns.PairGrid(df, y_vars=["EM"], x_vars=['Alpha', 'Beta', 'Gamma', 'W_1', 'W_2', 'W_3'], height=1.5)
g.map(sns.lineplot)
for ax in g.axes[-1,:]:
    xlabel = ax.xaxis.get_label_text()
    ax.xaxis.set_label_text(x_labels[xlabel])

plt.draw()
#plt.show()

plt.savefig("log/%s-%s_grid_search.png"%(dataset,sent_len))
plt.clf()
plt.close() 

import pandas as pd
import numpy as np
import itertools
import csv
import sys
import _ucrdtw
import statsmodels.api as sm
import cPickle
from statsmodels.tools import categorical
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from collections import OrderedDict, Counter
from utils import detect_peaks
from operator import add, mul


def closest_label(df,index):
    labels=[]
    events=np.array(df['label'][df['label']!='non-event'].index.tolist())
    labels_e=events[np.where(events<=index)[0]]
    try:
        if index in labels_e:
            labels=df['label'][index]
        else:
            labels=df['label'][np.max(labels_e)]
        return (labels.lower(),index-np.max(labels_e))
    except ValueError:
        return ('non-event',index)


def extract_lf(data):
    tmp=data[1]
    tmp['label'][pd.isnull(tmp['label'])]='non-event'
    ind = detect_peaks(tmp['a'],mph=30,mpd=17, show=False)
    temp=[]
    events=[]

    new_indices=[]
    labels=[]
    nobs=len(tmp['ax'])
    for index in ind:
        if index+8>nobs:
            pass
        elif index-8<0:
            pass
        else:
            label,dist=closest_label(tmp,tmp['ind'].iloc[index])
            new_indices.append([
                str(tmp['ind'].iloc[index]-8),
                str(tmp['ind'].iloc[index]+8),
                label,
                str(tmp['hand'].iloc[0]),
                str(dist)]) 
    return (new_indices)

def extract_indices(data):
    tmp=data
    start=tmp.index[0]
    events=[]
    for i in range(0,len(tmp['label'])-1):
        if tmp['label'].iloc[i]==tmp['label'].iloc[i+1]:
            pass
        elif tmp['label'].iloc[i]!=tmp['label'].iloc[i+1]:
            stop=tmp['ind'].iloc[i+1]
            events.append((start,stop,tmp['label'].iloc[i]))
            start=tmp['ind'].iloc[i]
        else:
            pass
    return events


def featurize(ts,dat,templates):
    """
    Get the features from the raw tennis data
    :param ts: numpy array of size (samples,sensor readings) which contains the raw data
    :param dat: tuple with start and stop indices to slice data
    :return: 
    :rtype : 
    """
    s=dat[0]
    t=dat[1]
    data=ts[s:t,:]
    length=1.0*len(data[:,0])
    a=np.sqrt(np.sum(np.array(data**2,dtype=float),axis=1))
    max_index=np.argmax(a)
    max_index=8
    if s-max_index < 0:
        s=16
    model1x=sm.OLS(range(1,9),ts[s+max_index:s+max_index+8,0])
    model1y=sm.OLS(range(1,9),ts[s+max_index:s+max_index+8,1]) 
    model1z=sm.OLS(range(1,9),ts[s+max_index:s+max_index+8,2]) 
    reg1=np.array([model1x.fit().params[0],model1y.fit().params[0],model1z.fit().params[0]])
    model2x=sm.OLS(range(1,9),ts[s+max_index-8:s+max_index,0])
    model2y=sm.OLS(range(1,9),ts[s+max_index-8:s+max_index,1]) 
    model2z=sm.OLS(range(1,9),ts[s+max_index-8:s+max_index,2]) 
    reg2=np.array([model2x.fit().params[0],model2y.fit().params[0],model2z.fit().params[0]])
    dist=np.zeros((len(templates)))
    for j,template in enumerate(templates):
        dist[j]=_ucrdtw.ucrdtw(data, template, 0.5, False)[1]
    correlation=np.corrcoef(np.c_[data,a],rowvar=0)
    return np.hstack([
    np.max(data,axis=0),\
    np.min(data,axis=0),np.mean(a),\
    np.std(a),\
    np.min(a),np.max(a),
    reg1,reg2,
    np.sign(ts[8,0:2]),
    (np.argmax(dat,axis=0)-np.argmin(dat,axis=0))/length,
    np.array([correlation[0,1],correlation[0,2],correlation[1,2],correlation[0,3],correlation[1,3],correlation[2,3]]),
    dist,
    np.sum(np.diff(np.sign(data),axis=0)>0,axis=0)/length,\
    np.sum(np.diff(np.sign(data),axis=0)<0,axis=0)/length])


if __name__ == '__main__':
    # Step 1: Create Event Labels 
    data=pd.read_csv('data/labeled_data.csv')
    data['a']=np.sqrt(data['ax']**2+data['ay']**2+data['az']**2)
    data.rename(columns={'wrist':'hand'}, inplace=True)
    data['hand'][data['hand']==True]='left'
    data['hand'][data['hand']==False]='right'
    data['ind']=data.index
    data['label'][pd.isnull(data['label'])]='non-event'
    all_data=data.groupby(['session'])
    input_data=[]
    for dataset in all_data:
        ind = detect_peaks(dataset[1]['a'],mph=30,mpd=17, show=False)
        if  len(ind)==0:
            pass
        else:
            input_data.append(extract_lf(dataset))  
    input_data=list(itertools.chain(*input_data))
    with open('data/labeled_events.csv', 'wb') as out:
       csv_out = csv.writer(out)
       for row in input_data:
           csv_out.writerow(row)     
    out.close()
    # Step 2: Read Event Labels 
    indices=pd.read_csv('data/labeled_events.csv').values.tolist()
    template=pd.read_csv('models/template_left.csv')
    templates=map(lambda x: np.array(x[1][['ax','ay','az']]),template.groupby('sample'))
    template_label=map(lambda x: x[1]['label'].iloc[0],template.groupby('sample'))
    # Step 3: Featurize Data
    features=[]
    labels=[]
    raw_data=[]
    for e in indices:
        if e[0]==e[1] or e[0]>e[1] or e[3]=='Lefty' or e[4]>10:
            pass
        else:
            
            if e[2]!='non-event':
                labels.append(e[2])
                features.append(featurize(np.array(data[['ax','ay','az','g1','g2','g3','gx','gy','gz','r1','r2','r3']]),e,templates))
                raw_data.append(np.array(data[['ax','ay','az']].iloc[e[0]:e[1],]))
            else:
                pass
    # Step 4: Classify Features         
    features=np.array(features)
    labels=np.array(labels)
    labels_c=categorical(np.array(labels),drop=True)
    labels_num=np.argmax(labels_c,axis=1)
    features_complete=features[~np.isnan(features).any(axis=1),:]
    labels_complete=labels[~np.isnan(features).any(axis=1)]
    f_train, f_test, l_train, l_test = train_test_split(features_complete, labels_complete, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=250,max_features=.33,oob_score=True)
    clf = clf.fit(f_train, l_train)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    y_predicted = clf.predict(f_test)
    y_pred=clf.predict(f_train)
    print "Classification report for classifier %s:\n%s\n" % (
        clf, metrics.classification_report(l_test, y_predicted))
    print "Confusion matrix:\n%s" % metrics.confusion_matrix(l_test, y_predicted)
    print "Classification report for classifier %s:\n%s\n" % (
        clf, metrics.classification_report(l_train, y_pred))
    print "Confusion matrix:\n%s" % metrics.confusion_matrix(l_train, y_pred)
    # Step 5: Save the classifier   
    with open('models/{0}.pkl'.format(sys.argv[1]), 'wb') as fid:
        cPickle.dump(clf, fid) 
    # Step 6: Explore Optimal Threshold Cutoff by Plotting Precision, Recall, and F1 over Potential Thresholds
    if sys.argv[2] == "plot":
        acc=[]
        peak_vals=range(20,50,2)
        pr=[]
        for val in peak_vals:
            for group in all_data:
                tp=0
                fp=0
                fn=0
                events=extract_indices(group[1])
                ind = detect_peaks(group[1]['a'],mph=val,mpd=17, show=False)
                ind = group[1]['ind'].iloc[ind].tolist()
                swings=filter(lambda x:x[2]!='non-event',events)
                nonev=filter(lambda x:x[2]=='non-event',events)
                if len(ind)==0:
                    pass
                else:
                    for index in ind:
                        label,distance=closest_label(group[1],index)
                        if(distance < 50):
                                tp+=1
                        elif(distance > 50):
                                fp+=1
                    fn=len(swings)-tp
                acc.append((tp,fp,fn))

            tp_sum=reduce(add,map(lambda x:x[0],acc))
            fp_sum=reduce(add,map(lambda x:x[1],acc))
            fn_sum=reduce(add,map(lambda x:x[2],acc)) 
            recall=tp_sum/(tp_sum+fn_sum)
            precision=tp_sum/(tp_sum+fp_sum)
            f1=2*recall*precision/(precision+recall)
            pr.append((precision,recall,f1))
            
        plt.plot(peak_vals,map(lambda x: x[0],pr),label='precision')
        plt.plot(peak_vals,map(lambda x: x[1],pr),label='recall')
        plt.plot(peak_vals,map(lambda x: x[2],pr),label='f-score')
        plt.legend()
        plt.show()
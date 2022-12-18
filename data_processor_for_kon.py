
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import json
import os
import random

import glob

queries = []


import collections

from transformers import DataProcessor, InputExample


logger = logging.getLogger(__name__)



class AppReviewProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass
    
    def get_train_examples(self, data_dir):
        """See base class."""
        path = r'data/allresult' 
        all_files = glob.glob(path + "/*.json")
        count_freq=[]

        queries=[]
        content_list=[]
        for filename in all_files:
            for line in open(filename, 'r'):
                jf=json.loads(line)
                queries.append(jf)
        
        examples1=[]
        examples2=[]
        examples3=[]
        unique=[]
        newqueries=[]

        for index, j in enumerate(queries):
            if j['content'] not in unique:
                unique.append(j['content'])
                newqueries.append(j)

        SEED =42  
        random.seed(SEED)
        random.shuffle(newqueries)
        
        for i,j in enumerate(newqueries):
            if j['annotation']!=None and len(j['annotation']['classificationResult'][0]['classes'])==3:
                x=j['annotation']['classificationResult'][0]['classes']
                guid = "%s-%s" % ("train", i)
                label=[]
                temp=j['annotation']['classificationResult'][0]['classes']
                if '全文有關核能' in j['annotation']['classificationResult'][0]['classes']:      
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文有關核能')])
                if '全文無關核能' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文無關核能')])
                if '全文支持核能' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文支持核能')])
                if '全文反對核能' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文反對核能')])
                if '全文無法判斷' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文無法判斷')])
                if '全文提及核廢料' in j['annotation']['classificationResult'][0]['classes']:   
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文提及核廢料')])
                if '全文無提及核廢料' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文無提及核廢料')])
                if label[0]=='全文有關核能'and label[1]=='全文無法判斷':
                    label[1]='全文有關核能無法判斷'
                if label[0]=='全文無關核能' and label[1]=='全文無法判斷':
                    label[1]='全文無關核能無法判斷'
                if label[0]=='全文無關核能'and label[1]=='全文支持核能':
                    continue
                elif label[0]=='全文無關核能'and label[1]=='全文反對核能':
                    continue
                elif label[0]=='全文無關核能'and label[1]=='全文有關核能無法判斷':
                    continue
                elif label[0]=='全文有關核能'and label[1]=='全文無關核能無法判斷':
                    continue  
                else:
                  # print(label)
                  # print(len(list(set(label))))
                  assert len(list(set(label)))==3
                  examples1.append(InputExample(guid=guid, text_a=j['content'], label=label[0]))
                  examples2.append(InputExample(guid=guid, text_a=j['content'], label=label[1]))
                  examples3.append(InputExample(guid=guid, text_a=j['content'], label=label[2]))
                  count_freq.extend((label[0],label[1],label[2]))
        counter=collections.Counter(count_freq)
        print(counter)
        examples1=examples1[:4500]
        examples2=examples2[:4500]
        examples3=examples3[:4500]
         
        return examples1,examples2,examples3


        
            
    def get_test_examples(self, data_dir):
        """See base class."""
        path = r'data/allresult' 
        all_files = glob.glob(path + "/*.json")
        queries=[]
        content=[]
        for filename in all_files:
            for line in open(filename, 'r'):   
                jf=json.loads(line)    
                queries.append(jf)
   
        examples1=[]
        examples2=[]
        examples3=[]
        unique=[]
        newqueries=[]
        for index, j in enumerate(queries):
            if j['content'] not in unique:
                unique.append(j['content'])
                newqueries.append(j)

        SEED =42   
        random.seed(SEED)
        random.shuffle(newqueries)
        for i,j in enumerate(newqueries):
            if j['annotation']!=None and len(j['annotation']['classificationResult'][0]['classes'])==3:
                x=j['annotation']['classificationResult'][0]['classes']
                guid = "%s-%s" % ("test", i)
                label=[]
                temp=j['annotation']['classificationResult'][0]['classes']
                if '全文有關核能' in j['annotation']['classificationResult'][0]['classes']:  
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文有關核能')])
                if '全文無關核能' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文無關核能')])
                if '全文支持核能' in j['annotation']['classificationResult'][0]['classes']:   
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文支持核能')])
                if '全文反對核能' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文反對核能')])
                if '全文無法判斷' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文無法判斷')])
                if '全文提及核廢料' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文提及核廢料')])
                if '全文無提及核廢料' in j['annotation']['classificationResult'][0]['classes']:
                    label.append(temp[j['annotation']['classificationResult'][0]['classes'].index('全文無提及核廢料')])
                if label[0]=='全文有關核能' and label[1]== '全文無法判斷':
                    label[1]='全文有關核能無法判斷'  
                if label[0]=='全文無關核能' and label[1]=='全文無法判斷':
                    label[1]='全文無關核能無法判斷'
                if label[0]=='全文無關核能'and label[1]=='全文支持核能':
                    continue
                elif label[0]=='全文無關核能'and label[1]=='全文反對核能':
                    continue
                elif label[0]=='全文無關核能'and label[1]=='全文有關核能無法判斷':
                    continue
                elif label[0]=='全文有關核能'and label[1]=='全文無關核能無法判斷':
                    continue  
                else:
                  examples1.append(InputExample(guid=guid, text_a=j['content'], label=label[0]))
                  examples2.append(InputExample(guid=guid, text_a=j['content'], label=label[1]))
                  examples3.append(InputExample(guid=guid, text_a=j['content'], label=label[2]))
                  content.append(j['content'])
        examples1=examples1[4500:]
        examples2=examples2[4500:]
        examples3=examples3[4500:]
        content=content[4500:]
        return examples1,examples2,examples3,content
    def get_labels(self):
        """See base class."""
        return ["全文有關核能", "全文無關核能"]
    def get_labels1(self):
        """See base class."""
        return ["全文支持核能", "全文反對核能", "全文有關核能無法判斷","全文無關核能無法判斷"]
    def get_labels2(self):
        """See base class."""
        return ["全文提及核廢料","全文無提及核廢料"]
appreview_processors = {

    "appreview": AppReviewProcessor,
}

appreview_output_modes = {
     
    "appreview": "classification",
}

appreview_tasks_num_labels = {

    "appreview":2,
    
}
appreview_tasks_num_labels2 = {

    "appreview":4,
    
}
appreview_tasks_num_labels3 = {

    "appreview":2,
    
}


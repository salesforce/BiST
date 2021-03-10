import numpy as np
from util import log

import os.path
import sys
import random
import h5py
import itertools

import pandas as pd
from data_util import data_util
import pickle as pkl

import torch 
import torch.utils.data as Data
from torch.autograd import Variable
import pdb 
import glob 
from tqdm import tqdm

__PATH__ = os.path.abspath(os.path.dirname(__file__))

def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

# PATHS
TYPE_TO_CSV = {'FrameQA': 'Train_frameqa_question.csv',
               'Count': 'Train_count_question.csv',
               'Trans': 'Train_transition_question.csv',
               'Action' : 'Train_action_question.csv'}
eos_word = '<EOS>'

class DatasetTGIF():
    '''
    API for TGIF dataset
    '''
    def __init__(self,
                 dataset_name='train',
                 fea_type=None, fea_path=None,
                 use_moredata=False,
                 max_n_videos=None,
                 data_type=None,
                 dataframe_dir=None,
                 vocab_dir=None,
                 is_test=0):
        self.dataframe_dir = dataframe_dir
        self.vocabulary_dir = vocab_dir
        self.use_moredata = use_moredata
        self.dataset_name = dataset_name
        self.fea_type=fea_type
        self.fea_path=fea_path
        self.max_n_videos = max_n_videos
        self.data_type = data_type
        self.data_df = self.read_df_from_csvfile()
        self.GLOVE_EMBEDDING_SIZE = 300
        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]
        self.ids = list(self.data_df.index)
        if dataset_name == 'train':
            random.shuffle(self.ids)
        self.fea = self.read_tgif_fea()
        self.is_test = is_test

    def __len__(self):
        if self.max_n_videos is not None:
            if self.max_n_videos <= len(self.ids):
                return self.max_n_videos
        return len(self.ids)

    def read_tgif_fea(self):
        out = []
        if self.fea_type is not None and self.fea_type[0] != 'none':
            for ftype in self.fea_type:
                if ftype == 'none': 
                    out.append(None)
                    continue
                basepath = self.fea_path.replace('<FeaType>', ftype)
                files = [f for f in glob.glob('/'.join(basepath.split('/')[:-1]) + "/*.npy")]
                features = {}
                for filepath in tqdm(files, total=len(files)):
                    vid = filepath.split('/')[-1].split('.')[0]
                    features[vid] = (filepath)
                out.append(features)
        else:
            out = None 
        return out 
   
    def read_df_from_csvfile(self):
        assert self.data_type in ['FrameQA', 'Count', 'Trans', 'Action'], 'Should choose data type '

        if self.data_type == 'FrameQA':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_frameqa_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_frameqa_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_frameqa_question.csv'), sep='\t')
        elif self.data_type == 'Count':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_count_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_count_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_count_question.csv'), sep='\t')
        elif self.data_type == 'Trans':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_transition_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_transition_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_transition_question.csv'), sep='\t')
        elif self.data_type == 'Action':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_action_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_action_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_action_question.csv'), sep='\t')
            
        assert_exists(train_data_path)
        assert_exists(test_data_path)

        if self.dataset_name == 'train':
            data_df = pd.read_csv(train_data_path, sep='\t')
        elif self.dataset_name == 'test':
            data_df = pd.read_csv(test_data_path, sep='\t')

        data_df = data_df.set_index('vid_id')
        data_df['row_index'] = range(1, len(data_df)+1) # assign csv row index
        return data_df

    @property
    def n_words(self):
        ''' The dictionary size. '''
        if not hasattr(self, 'word2idx'):
            raise Exception('Dictionary not built yet!')
        return len(self.word2idx)

    def __repr__(self):
        if hasattr(self, 'word2idx'):
            return '<Dataset (%s) with %d videos and %d words>' % (
                self.dataset_name, len(self), len(self.word2idx))
        else:
            return '<Dataset (%s) with %d videos -- dictionary not built>' % (
                self.dataset_name, len(self))

    def split_sentence_into_words(self, sentence, eos=True):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        When tokenizing, I use ``data_util.clean_str``
        '''
        try:
            words = data_util.clean_str(sentence).split()
        except:
            print(sentence)
            sys.exit()
        if eos:
            words = words + [eos_word]
        for w in words:
            if not w:
                continue
            yield w
        
    def build_word_vocabulary(self, all_captions_source=None,
                              word_count_threshold=0,):
        '''
        borrowed this implementation from @karpathy's neuraltalk.
        '''
        log.infov('Building word vocabulary (%s) ...', self.dataset_name)

        if all_captions_source is None:
            all_captions_source = self.get_all_captions()

        # enumerate all sentences to build frequency table
        word_counts = {}
        nsents = 0
        for sentence in all_captions_source:
            nsents += 1
            for w in self.split_sentence_into_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1


        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        log.info("Filtered vocab words (threshold = %d), from %d to %d",
                 word_count_threshold, len(word_counts), len(vocab))

        # build index and vocabularies
        self.word2idx = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
        self.idx2word = {0:'<unk>', 1:'<blank>', 2:'<sos>', 3:'<eos>'}
        
        for idx, w in enumerate(vocab, start=4):
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        import pickle as pkl
        pkl.dump(self.word2idx, open(os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl'%self.data_type), 'wb'))
        pkl.dump(self.idx2word, open(os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl'%self.data_type), 'wb'))

        answers = list(set(self.total_q['answer'].values))
        self.ans2idx = {}
        self.idx2ans = {}
        for idx, w in enumerate(answers):
            self.ans2idx[w]=idx
            self.idx2ans[idx]=w
        pkl.dump(self.ans2idx, open(os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.data_type), 'wb'))
        pkl.dump(self.idx2ans, open(os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.data_type), 'wb'))

    def load_word_vocabulary(self):
        word_matrix_path = os.path.join(self.vocabulary_dir, 'vocab_embedding_%s.pkl'%self.data_type)

        word2idx_path = os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl'%self.data_type)
        idx2word_path = os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl'%self.data_type)
        ans2idx_path = os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.data_type)
        idx2ans_path = os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.data_type)
                
        if not (os.path.exists(word2idx_path) and os.path.exists(idx2word_path) \
               and os.path.exists(ans2idx_path) and os.path.exists(idx2ans_path)):
            self.build_word_vocabulary()

        with open(word2idx_path, 'rb') as f:
            self.word2idx = pkl.load(f, encoding='latin1')
        log.info("Load word2idx from pkl file : %s", word2idx_path)

        with open(idx2word_path, 'rb') as f:
            self.idx2word = pkl.load(f, encoding='latin1')
        log.info("Load idx2word from pkl file : %s", idx2word_path)

        with open(ans2idx_path, 'rb') as f:
            self.ans2idx = pkl.load(f, encoding='latin1')
        log.info("Load answer2idx from pkl file : %s", ans2idx_path)

        with open(idx2ans_path, 'rb') as f:
            self.idx2ans = pkl.load(f, encoding='latin1')
        log.info("Load idx2answers from pkl file : %s", idx2ans_path)


    def share_word_vocabulary_from(self, dataset):
        assert hasattr(dataset, 'idx2word') and hasattr(dataset, 'word2idx'), \
            'The dataset instance should have idx2word and word2idx'
        assert (isinstance(dataset.idx2word, dict) or isinstance(dataset.idx2word, list)) \
                and isinstance(dataset.word2idx, dict), \
            'The dataset instance should have idx2word and word2idx (as dict)'

        if hasattr(self, 'word2idx'):
            log.warn("Overriding %s' word vocabulary from %s ...", self, dataset)

        self.idx2word = dataset.idx2word
        self.word2idx = dataset.word2idx
        self.ans2idx = dataset.ans2idx
        self.idx2ans = dataset.idx2ans
        if hasattr(dataset, 'word_matrix'):
            self.word_matrix = dataset.word_matrix


    # Dataset Access APIs (batch loading, sentence etc)
    def iter_ids(self, shuffle=False):

        #if self.data_type == 'Trans':
        if shuffle:
            random.shuffle(self.ids)
        for key in self.ids:
            yield key

    def get_all_captions(self):
        '''
        Iterate caption strings associated in the vid/gifs.
        '''
        qa_data_df = pd.read_csv(os.path.join(self.dataframe_dir, TYPE_TO_CSV[self.data_type]), sep='\t')

        all_sents = []
        for row in qa_data_df.iterrows():
            all_sents.extend(self.get_captions(row))
        self.data_type
        return all_sents

    def get_captions(self, row):
        if self.data_type == 'FrameQA':
            columns = ['description', 'question', 'answer']
        elif self.data_type == 'Count':
            columns = ['question']
        elif self.data_type == 'Trans':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        elif self.data_type == 'Action':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']

        sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
        return sents

    def load_video_feature(self, key):
        key_df = self.data_df.loc[key,'key']
        video_id = str(key_df)

        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['pool5', 'res5c']
            video_feature = np.array(self.feat_h5[video_id])
            if self.layer.lower() == 'res5c':
                video_feature = np.transpose(
                    video_feature.reshape([-1,2048,7,7]), [0, 2, 3, 1])
                assert list(video_feature.shape[1:]) == [7, 7, 2048]
            elif self.layer.lower() == 'pool5':
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 2048]

        elif self.image_feature_net.lower() == 'c3d':
            assert self.layer.lower() in ['fc6', 'conv5b']
            video_feature = np.array(self.feat_h5[video_id])

            if self.layer.lower() == 'fc6':
                if len(video_feature.shape) == 1:
                    video_feature = np.expand_dims(video_feature, axis=0)
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 4096]
            elif self.layer.lower() == 'conv5b':
                if len(video_feature.shape) == 4:
                    video_feature = np.expand_dims(video_feature, axis=0)
                # we use resnet C3D
                video_feature = np.transpose(
                    video_feature.reshape([-1,2048,7,7]), [0,2,3,1])
                assert list(video_feature.shape[1:]) == [7, 7, 2048]
                

        elif self.image_feature_net.lower() == 'concat':
            assert self.layer.lower() in ['fc', 'conv']
            c3d_feature = np.array(self.feat_h5["c3d"][video_id])
            resnet_feature = np.array(self.feat_h5["resnet"][video_id])
            if len(c3d_feature.shape) == 1:
                c3d_feature = np.expand_dims(c3d_feature, axis=0)

            if not len(c3d_feature) == len(resnet_feature):
                max_len = min(len(c3d_feature),len(resnet_feature))
                c3d_feature = c3d_feature[:max_len]
                resnet_feature = resnet_feature[:max_len]

            if self.layer.lower() == 'fc':
                video_feature = np.concatenate((c3d_feature, resnet_feature),
                                                axis=len(c3d_feature.shape)-1)
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 4096+2048]
            elif self.layer.lower() == 'conv':
                c3d_feature = np.transpose(c3d_feature.reshape([-1,2048,7,7]), [0,2,3,1]) # we use resnet C3D
                resnet_feature = np.transpose(resnet_feature.reshape([-1,2048,7,7]), [0, 2, 3, 1])
                video_feature = np.concatenate((c3d_feature, resnet_feature),
                                               axis=len(c3d_feature.shape)-1)
                assert list(video_feature.shape[1:]) == [7, 7, 2048+2048]  # we use resnet C3D

        return video_feature

    def get_video_feature_dimension(self):
        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['fc1000', 'pool5', 'res5c']
            if self.layer.lower() == 'res5c':
                return (self.max_length, 7, 7, 2048)
            elif self.layer.lower() == 'pool5':
                return (self.max_length, 1, 1, 2048)
        elif self.image_feature_net.lower() == 'c3d':
            if self.layer.lower() == 'fc6':
                return (self.max_length, 1, 1, 4096)
            elif self.layer.lower() == 'conv5b':
                return (self.max_length, 7, 7, 2048)
        elif self.image_feature_net.lower() == 'concat':
            assert self.layer.lower() in ['fc', 'conv']
            if self.layer.lower() == 'fc':
                return (self.max_length, 1, 1, 4096+2048)
            elif self.layer.lower() == 'conv':
                return (self.max_length, 7, 7, 2048+2048)

    def get_video_feature(self, key):
        video_feature = self.load_video_feature(key)
        return video_feature

    def convert_sentence_to_matrix(self, sentence, eos=True):
        '''
        Convert the given sentence into word indices and masks.
        WARNING: Unknown words (not in vocabulary) are revmoed.

        Args:
            sentence: A str for unnormalized sentence, containing T words

        Returns:
            sentence_word_indices : list of (at most) length T,
                each being a word index
        '''
        sent2indices = [self.word2idx[w] if w in self.word2idx else 0 for w in
                        self.split_sentence_into_words(sentence,eos)] # 1 is UNK, unknown
        return np.array(sent2indices, dtype=np.int32)

    def get_video_mask(self, video_feature):
        video_length = video_feature.shape[0]
        return data_util.fill_mask(self.max_length, video_length, zero_location='LEFT')

    def get_sentence_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='RIGHT')

    def get_question(self, key):
        '''
        Return question string for given key.
        '''
        question = self.data_df.loc[key, ['question','description']].values
        if len(list(question.shape)) > 1:
            question = question[0]
        question = question[0]
        return self.convert_sentence_to_matrix(question, eos=False)

    def get_question_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='LEFT')

    def get_answer(self, key):
        answer = self.data_df.loc[key, ['answer','type']].values

        if len(list(answer.shape)) > 1:
            answer = answer[0]

        anstype = answer[1]
        answer = answer[0]

        return answer, anstype

    def get_FrameQA_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros(
            [batch_size] + list(self.get_video_feature_dimension()), dtype=np.float32)

        # Question, Right most aligned
        batch_question = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_right = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_video_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)
        batch_answer = np.zeros([batch_size, 1])
        batch_answer_type = np.zeros([batch_size, 1])
        questions = []

        video_lengths = []
        question_lengths = []
        
        for k in range(batch_size):
            key = chunk[k]
            video_feature = self.get_video_feature(key)
            video_mask = self.get_video_mask(video_feature)

            vl = min(self.max_length, video_feature.shape[0])
            video_lengths.append(vl)
            
            batch_video_feature_convmap[k, :] = data_util.pad_video(
                video_feature, self.get_video_feature_dimension())
            batch_video_mask[k] = video_mask

            answer, answer_type = self.get_answer(key)
            if str(answer) in self.ans2idx:
                answer = self.ans2idx[answer]
            else:
                # unknown token, check later
                answer = 1
            question = self.get_question(key)            
            question_mask = self.get_question_mask(question)
            #print('1----------------',question,len(question))
            question_lengths.append(len(question))
            
            # Left align
            batch_question[k, :len(question)] = question
            # Right align
            batch_question_right[k, -len(question):] = question
            batch_question_mask[k] = question_mask
            question_pad = np.zeros([self.max_length])
            question_pad[:len(question)] = question
            questions.append(question_pad)
            batch_answer[k] = answer
            batch_answer_type[k] = float(int(answer_type))
            batch_debug_sent[k] = self.data_df.loc[key, 'question']

        ret = {
            'ids': chunk,
            'video_lengths': video_lengths,
            'video_features': batch_video_feature_convmap,
            'question_words': batch_question,
            'question_words_right': batch_question_right,
            'question_lengths': question_lengths,
            'video_mask': batch_video_mask,
            'question_mask': batch_question_mask,
            'answer': batch_answer,
            'answer_type': batch_answer_type,
            'debug_sent': batch_debug_sent
        }
        return ret

    def get_Count_question(self, key):
        '''
        Return question string for given key.
        '''
        question = self.data_df.loc[key, 'question']
        return self.convert_sentence_to_matrix(question, eos=False)

    def get_Count_question_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='RIGHT')

    def get_Count_answer(self, key):
        return self.data_df.loc[key, 'answer']

    def get_Count_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros(
            [batch_size] + list(self.get_video_feature_dimension()), dtype=np.float32)

        # Question, Right most aligned
        batch_question = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_right = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_video_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_question_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)
        batch_answer = np.zeros([batch_size, 1])
        
        video_lengths = []
        question_lengths = []
        
        for k in range(batch_size):
            key = chunk[k]
            video_feature = self.get_video_feature(key)
            video_mask = self.get_video_mask(video_feature)
            
            vl = min(self.max_length, video_feature.shape[0])
            video_lengths.append(vl)

            batch_video_feature_convmap[k, :] = data_util.pad_video(
                video_feature, self.get_video_feature_dimension())
            batch_video_mask[k] = video_mask

            answer = max(self.get_Count_answer(key), 1)

            question = self.get_Count_question(key)
            question_mask = self.get_Count_question_mask(question)
            # Left align
            batch_question[k, :len(question)] = question
            # Right align
            batch_question_right[k, -len(question):] = question
            batch_question_mask[k] = question_mask
            question_lengths.append(len(question))
            batch_answer[k] = answer
            batch_debug_sent[k] = self.data_df.loc[key, 'question']

        ret = {
            'ids': chunk,
            'video_lengths': video_lengths,
            'video_features': batch_video_feature_convmap,
            'question_words': batch_question,
            'question_words_right': batch_question_right,
            'question_lengths': question_lengths,
            'video_mask': batch_video_mask,
            'question_mask': batch_question_mask,
            'answer': batch_answer,
            'debug_sent': batch_debug_sent
        }
        return ret

    def get_Trans_dict(self, key):
        a1 = self.data_df.loc[key, 'a1'].strip()
        a2 = self.data_df.loc[key, 'a2'].strip()
        a3 = self.data_df.loc[key, 'a3'].strip()
        a4 = self.data_df.loc[key, 'a4'].strip()
        a5 = self.data_df.loc[key, 'a5'].strip()
        question = self.data_df.loc[key, 'question'].strip()
        row_index = self.data_df.loc[key, 'row_index']

        # as list of sentence strings
        candidates = [a1, a2, a3, a4, a5]
        answer = self.data_df.loc[key, 'answer']

        candidates_to_indices = [self.convert_sentence_to_matrix(question + ' ' + x)
                                 for x in candidates]

        return {
            'answer' : answer,
            'candidates': candidates_to_indices,
            'raw_sentences': candidates,
            'row_indices' : row_index,
            'question' : question
        }

    def get_Trans_matrix(self, candidates, is_left=True):
        candidates_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            sentence = candidates[k]
            if is_left:
                candidates_matrix[k, :len(sentence)] = sentence
            else:
                candidates_matrix[k, -len(sentence):] = sentence
        return candidates_matrix

    def get_Trans_mask(self, candidates):
        mask_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            mask_matrix[k] = data_util.fill_mask(self.max_length,
                                                 len(candidates[k]),
                                                 zero_location='RIGHT')
        return mask_matrix

    def get_Trans_result(self, chunk):
        batch_size = len(chunk)
        batch_video_feature_convmap = np.zeros(
            [batch_size] + list(self.get_video_feature_dimension()), dtype=np.float32)

        batch_candidates = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)
        batch_candidates_right = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)
        batch_answer = np.zeros([batch_size], dtype=np.uint32)

        batch_video_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_candidates_mask = np.zeros([batch_size, 5, self.max_length], dtype=np.uint32)

        batch_debug_sent = np.asarray([None] * batch_size)
        batch_raw_sentences = np.asarray([[None]*5 for _ in range(batch_size)]) # [batch_size, 5]
        batch_row_indices = np.asarray([-1] * batch_size)

        video_lengths = []
        candidate_lengths = []
        batch_questions = []
        question_word_nums = []
        
        for k in range(batch_size):
            key = chunk[k]

            MC_dict = self.get_Trans_dict(key)
            candidates = MC_dict['candidates']
            
            question = MC_dict['question']

            qnum = len(question.split())
            question_word_nums.append(qnum)
            raw_sentences = MC_dict['raw_sentences']
            
            answer = int(MC_dict['answer'])    # a choice from 0-4, since there are five candidate answers
            question = MC_dict['question']
            
            video_feature = self.get_video_feature(key)
            candidates_matrix = self.get_Trans_matrix(candidates)
            candidates_matrix_right = self.get_Trans_matrix(candidates, is_left=False)

            vl = min(self.max_length, video_feature.shape[0])
            video_lengths.append(vl)
            
            # get candidate length
            cand_lens = []
            for cand in candidates:
                vl = min(self.max_length, len(cand))
                cand_lens.append(vl)
            candidate_lengths.append(cand_lens)
            
            video_mask = self.get_video_mask(video_feature)
            candidates_mask = self.get_Trans_mask(candidates)

            batch_video_feature_convmap[k, :] = data_util.pad_video(
                video_feature, self.get_video_feature_dimension())

            batch_candidates[k] = candidates_matrix
            batch_candidates_right[k] = candidates_matrix_right
            batch_raw_sentences[k, :] = raw_sentences
            batch_answer[k] = answer
            batch_video_mask[k] = video_mask
            batch_candidates_mask[k] = candidates_mask
            batch_row_indices[k] = MC_dict['row_indices']
            batch_questions.append(question)

            batch_debug_sent[k] = self.data_df.loc[key, 'a'+str(int(answer+1))]

        ret = {
            'ids': chunk,
            'video_lengths': video_lengths,
            'video_features': batch_video_feature_convmap,
            'candidates': batch_candidates,
            'candidates_right': batch_candidates_right,
            'candidate_lengths': candidate_lengths,
            'answer': batch_answer,
            'raw_sentences': batch_raw_sentences,
            'video_mask': batch_video_mask,
            'candidates_mask': batch_candidates_mask,
            'debug_sent': batch_debug_sent,
            'row_indices' : batch_row_indices,
            'question': batch_questions,
            'num_mult_choices':5,
            'question_word_nums':question_word_nums,
        }
        return ret

    def get_Action_result(self, chunk):
        return self.get_Trans_result(chunk)

    def next_batch(self, batch_size=64, include_extra=False, shuffle=True):
        if not hasattr(self, '_batch_it'):
            self._batch_it = itertools.cycle(self.iter_ids(shuffle=shuffle))

        chunk = []
        for k in range(batch_size):
            key = next(self._batch_it)
            chunk.append(key)
        if self.data_type == 'FrameQA':
            return self.get_FrameQA_result(chunk)
        # Make custom function to make batch!
        elif self.data_type == 'Count':
            return self.get_Count_result(chunk)
        elif self.data_type == 'Trans':
            return self.get_Trans_result(chunk)
        elif self.data_type == 'Action':
            return self.get_Action_result(chunk)
        else:
            raise Exception('data_type error in next_batch')

    def batch_iter(self, num_epochs, batch_size, shuffle=True):
        for epoch in range(num_epochs):
            steps_in_epoch = int(len(self) / batch_size)

            for s in range(steps_in_epoch+1):
                yield self.next_batch(batch_size, shuffle=shuffle)

    def split_dataset(self, ratio=0.1):
        data_split = DatasetTGIF(dataset_name=self.dataset_name,
                                 fea_type=self.fea_type, fea_path=self.fea_path,
                                 use_moredata=self.use_moredata,
                                 max_n_videos=self.max_n_videos,
                                 data_type=self.data_type,
                                 dataframe_dir=self.dataframe_dir,
                                 vocab_dir=self.vocabulary_dir,
                                 is_test=self.is_test)

        data_split.ids = self.ids[-int(ratio*len(self.ids)):]
        self.ids = self.ids[:-int(ratio*len(self.ids))]
        return data_split

class Dataset(Data.Dataset):
    def __init__(self, data_info):
        self.vid = data_info['vid']
        self.qa_id = data_info['qa_id']
        self.plain_question = data_info['plain_question']
        self.question = data_info['question']
        self.answer = data_info['answer']
        self.fea_path = data_info['fea_path']
        self.num_total_seqs = len(data_info['qa_id'])
        if 'plain_candidates' in data_info:
            self.plain_candidates = data_info['plain_candidates']

    def __getitem__(self, index): 
        item_info = {
            'vid':self.vid[index], 
            'qa_id': self.qa_id[index],
            'plain_question': self.plain_question[index], 
            'question': self.question[index],
            'answer': self.answer[index],
            'fea_path': self.fea_path[index],
            'plain_candidates': self.plain_candidates[index]
            }
        return item_info
    
    def __len__(self):
        return self.num_total_seqs    

class Batch:
    def __init__(self, query, fts, trg, pad, vids, qa_ids, plain_questions, plain_candidates, cuda=False):
        self.vids = vids
        self.qa_ids = qa_ids
        self.plain_questions = plain_questions
        self.plain_candidates = plain_candidates 
        self.cuda = cuda
        self.query = self.to_cuda(query)
        self.query_mask = self.to_cuda((query != pad).unsqueeze(-2))        
        self.temporal_ft = None
        self.spatial_ft = None 
        self.fts = None 
        self.spatial_mask = None 
        self.temporal_mask = None
        self.fts_mask = None 
        if type(fts) == torch.Tensor:
            self.fts = self.to_cuda(fts.float())
        else:
            self.fts = self.to_cuda(torch.from_numpy(fts).float())
        self.spatial_mask = self.to_cuda((self.fts.sum(1).sum(-1)!=0).unsqueeze(-2))
        self.temporal_mask = self.to_cuda((self.fts.sum(2).sum(-1)!=0).unsqueeze(-2))
       
        self.trg = self.to_cuda(trg)
        self.trg_mask = None 
        self.qntokens = (self.query != pad).data.sum()
        self.ntokens = self.trg.shape[0]
               
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask    

    def move_to_cuda(self):
        self.query = self.query.to('cuda', non_blocking=True)
        self.query_mask = self.query_mask.to('cuda', non_blocking=True)
        if self.fts is not None:
            self.fts = self.fts.to('cuda', non_blocking=True)
            if self.fts_mask is not None:
                self.fts_mask = self.fts_mask.to('cuda', non_blocking=True)
            else:
                self.spatial_mask = self.spatial_mask.to('cuda', non_blocking=True)
                self.temporal_mask = self.temporal_mask.to('cuda', non_blocking=True)        
        self.trg = self.trg.to('cuda', non_blocking=True)
    
    def to_cuda(self, tensor):
        if self.cuda: return tensor.cuda()
        return tensor 
    
def collate_fn(data):
    def pad_seq(seqs, pad_token):
        max_length = max([len(s) for s in seqs])
        output = []
        for seq in seqs:
            result = np.ones(max_length, dtype=seq.dtype)*pad_token
            result[:seq.shape[0]] = seq 
            output.append(result)
        return output 

    def prepare_data(seqs):
        return torch.from_numpy(np.asarray(seqs)).long()
            
    def load_np(filepath):
        feature = np.load(filepath, allow_pickle=True)
        if len(feature.shape) == 2:
            return feature
        else:
            return np.transpose(feature.reshape(feature.shape[0], feature.shape[1], -1), (0,2,1))

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    # assume only one feature type
    fea_ls = [load_np(fp[0]) for fp in item_info['fea_path']]
    x_len = max([i.shape[0] for i in fea_ls]) 
    n_seqs = len(fea_ls)
    if len(fea_ls[0].shape) == 3: 
        x_batch = np.zeros((n_seqs, x_len, 
                            fea_ls[0].shape[1], 
                            fea_ls[0].shape[2]),dtype=np.float32)        
    else:
        x_batch = np.zeros((n_seqs, x_len, fea_ls[0].shape[-1]),dtype=np.float32)
    for j, fea in enumerate(fea_ls):
        x_batch[j, :len(fea)] = fea

    if len(item_info['plain_candidates'])>0 and type(item_info['plain_candidates'][0])==list: 
        q_batch = []
        for q in item_info['question']:
            q_batch.extend(q)
        q_batch = prepare_data(pad_seq(q_batch, 1))
        x_batch = torch.tensor(x_batch).unsqueeze(1).expand(x_batch.shape[0], 5, x_batch.shape[1], x_batch.shape[2], x_batch.shape[3])
        x_batch = x_batch.reshape(-1, x_batch.shape[-3], x_batch.shape[-2], x_batch.shape[-1])
    else:    
        q_batch = prepare_data(pad_seq(item_info['question'], 1))
    a_batch = prepare_data(item_info['answer'])
    batch = Batch(q_batch, x_batch, a_batch, 1, item_info['vid'], item_info['qa_id'], item_info['plain_question'], item_info['plain_candidates'])

    return batch

def get_count_data(data, nb_samples=100):
    out = {}
    if data.is_test:
        out['qa_id'] = data.ids[:nb_samples]
    else:
        out['qa_id'] = data.ids #[:nb_samples]
    out['vid'] = []
    out['plain_question'] = []
    out['question'] = []
    out['plain_candidates'] = []
    out['answer'] = []
    out['fea_path'] = []
    for idx in tqdm(out['qa_id'], total=len(out['qa_id'])):
        row = data.data_df[data.data_df.index == idx]
        out['vid'].append(row.gif_name[0])
        question = row.question[0]
        out['plain_question'].append(question)
        enc_question = data.convert_sentence_to_matrix(question, eos=False)
        out['question'].append(enc_question)
        answer = max(row.answer[0], 1)
        out['answer'].append(answer)
        out['plain_candidates'].append(answer)
        if data.is_test:
            out['fea_path'].append([fea['1'] for fea in data.fea]) 
        else:
            out['fea_path'].append([fea[row.gif_name[0]] for fea in data.fea])
    return out 

def get_trans_data(data, nb_samples=100):
    out = {}
    if data.is_test:
        out['qa_id'] = data.ids[:nb_samples]
    else:
        out['qa_id'] = data.ids #[:nb_samples]
    out['vid'] = []
    out['plain_question'] = []
    out['question'] = []
    out['plain_candidates'] = []
    out['answer'] = []
    out['fea_path'] = []
    for idx in tqdm(out['qa_id'], total=len(out['qa_id'])):
        row = data.data_df[data.data_df.index == idx]
        out['vid'].append(row.gif_name[0])
        question = row.question[0]
        out['plain_question'].append(question)
        a1 = row.a1[0].strip()
        a2 = row.a2[0].strip()
        a3 = row.a3[0].strip()
        a4 = row.a4[0].strip()
        a5 = row.a5[0].strip()
        candidates = [a1, a2, a3, a4, a5]    
        out['plain_candidates'].append(candidates)
        enc_question = [data.convert_sentence_to_matrix(question + ' ' + x, eos=False) for x in candidates]
        out['question'].append(enc_question)
        answer = row.answer[0]
        out['answer'].append(answer)       
        if data.is_test:
            out['fea_path'].append([fea['3'] for fea in data.fea])
        else: 
            out['fea_path'].append([fea[row.gif_name[0]] for fea in data.fea])
    return out 

def get_frameqa_data(data, nb_samples=100):
    out = {}
    if data.is_test:
        out['qa_id'] = data.ids[:nb_samples]
    else:    
        out['qa_id'] = data.ids #[:nb_samples]
    out['vid'] = []
    out['plain_question'] = []
    out['question'] = []
    out['plain_candidates'] = []
    out['answer'] = []
    out['fea_path'] = []
    for idx in tqdm(out['qa_id'], total=len(out['qa_id'])):
        row = data.data_df[data.data_df.index == idx]
        out['vid'].append(row.gif_name[0])
        question = row.question[0]
        out['plain_question'].append(question)
        enc_question = data.convert_sentence_to_matrix(question, eos=False)
        out['question'].append(enc_question)
        answer = row.answer[0]
        out['plain_candidates'].append(answer) 
        out['answer'].append(data.ans2idx[answer])
        if data.is_test:
            out['fea_path'].append([fea['4'] for fea in data.fea])
        else:
            out['fea_path'].append([fea[row.gif_name[0]] for fea in data.fea])
    return out 

def create_dataloader(data, batch_size, shuffle, args, num_workers=0):
    if args.task == 'Count':
        out = get_count_data(data)
    elif args.task in ['Trans', 'Action']:
        out = get_trans_data(data)
    elif args.task in ['FrameQA']:
        out = get_frameqa_data(data)
    dataset = Dataset(out)     
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  pin_memory=True)
    return data_loader, len(out['qa_id'])



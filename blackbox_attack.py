import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
from data import part_pytorch_dataset
from model import *

class blackbox_attack:

    def __init__(self,membership_attack_number,name='baseline_attack',num_classes=10):

        np.random.seed(seed = 12345)
        np.set_printoptions(suppress=True)

        self.name = name
        self.num_classes = num_classes
        self.membership_attack_number = membership_attack_number


    def attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels,avg_loss=0):

        if (self.name == 'baseline'):
            self._baseline_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)
        if (self.name == 'avg_loss'):
            self.avg_loss = avg_loss
            self._avg_loss_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)
        if (self.name == 'top1'):
            self._top1_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)
        if (self.name == 'top3'):
            self._top3_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)
        if (self.name == 'global_prob'):
            self._global_prob_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)
        if (self.name == 'per_class'):
            self._per_class_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)
        if (self.name == 'instance_distance'):
            self._instance_distance_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)
        if (self.name == 'instance_prob'):
            self._instance_prob_attack(shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels)


    def _baseline_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):

        reshaped_classes = test_classes
        reshaped_classes = np.reshape(reshaped_classes, (-1))

        features = np.reshape(test_confidence, (len(reshaped_classes),-1))
        features = np.argmax(features, axis=1)
        corr = [(features[i] == reshaped_classes[i]) for i in range(len(reshaped_classes))]
        corr = np.array(corr)
        corr = np.reshape(corr,(-1))
        labels = np.reshape(test_labels,(-1))
        acc = 0

        #print (reshaped_classes.shape,features.shape,corr.shape,labels.shape)


        for i in range(len(reshaped_classes)):
            if (corr[i] == 1 and labels[i] == 1):
                acc += 1
            if (corr[i] == 0 and labels[i] == 0):
                acc += 1

        acc = acc / len(reshaped_classes)
        print ("baseline attack acc = %.2f" % (acc*100))
        #print (classification_report(labels, corr))

    def _avg_loss_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):

        reshaped_classes = (test_classes.copy()).astype(np.int64)
        reshaped_classes = np.reshape(reshaped_classes, (-1))
        total_num = len(reshaped_classes)
        features = np.reshape(test_confidence, (total_num,self.num_classes))
        features = [features[i, reshaped_classes[i]] for i in range(features.shape[0])]
        features = np.array(features)
        features = np.log(features) * -1
        features = np.nan_to_num(features)
        labels = np.reshape(test_labels,(-1))
        # shadow_avg_loss = np.average(total_avg_loss)/(args.target_data_size*args.model_number/2)
        #shadow_avg_loss = np.average(total_avg_loss)
        shadow_avg_loss = self.avg_loss
        print ("shadow avg loss = %.2f " % (shadow_avg_loss))
        corr = 0
        predict = np.zeros((len(labels)))
        for i in range(len(labels)):
            predict[i] = (shadow_avg_loss > features[i])
            if (predict[i] == 1 and labels[i] == 1):
                corr += 1
            if (predict[i] == 0 and labels[i] == 0):
                corr += 1
        print ("global avg loss accuracy %.2f" % (corr*100 / len(labels)))
        #print (classification_report(labels, predict))

    def _top1_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):

        total_shadow_num = len(np.reshape(shadow_classes,-1))
        total_test_num = len(np.reshape(test_classes,-1))
        shadow_confidence = np.reshape(shadow_confidence,(total_shadow_num,-1))
        test_confidence = np.reshape(test_confidence,(total_test_num,-1))
        train_features = np.amax(shadow_confidence, axis=1)
        train_features = np.reshape(train_features, (-1, 1))
        test_features = np.amax(test_confidence, axis=1)
        test_features = np.reshape(test_features, (-1, 1))
        train_features = np.nan_to_num(train_features)
        test_features = np.nan_to_num(test_features)
        train_labels = np.reshape(shadow_labels,(-1))
        test_labels = np.reshape(test_labels,(-1))

        model = LogisticRegression(random_state=0, solver='lbfgs')
        model.fit(train_features, train_labels)
        print ("global highest lr accuracy = %f " % (model.score(test_features, test_labels)*100))
        #print (classification_report(test_labels, model.predict(test_features)))

    def _top3_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):

        total_shadow_num = len(np.reshape(shadow_classes,-1))
        total_test_num = len(np.reshape(test_classes,-1))

        train_features = np.reshape(shadow_confidence,(total_shadow_num,-1))
        train_features = np.sort(train_features, axis=1)
        train_features = train_features[:,-3:]

        test_features = np.reshape(test_confidence,(total_test_num,-1))
        test_features = np.sort(test_features, axis=1)
        test_features = test_features[:,-3:]

        train_features = np.nan_to_num(train_features)
        test_features = np.nan_to_num(test_features)
        train_labels = np.reshape(shadow_labels,(-1))
        test_labels = np.reshape(test_labels,(-1))

        model = LogisticRegression(random_state=0, solver='lbfgs')
        model.fit(train_features, train_labels)
        print ("global top3 lr accuracy = %f " % (model.score(test_features, test_labels)*100))
        #y_pred = model.predict(test_features)
        #print (classification_report(test_labels, y_pred))

        ### NN top3 attack
        ### similar acc as LR top3 attack
        nn_top3_acc = self._nn_attack(train_features, train_labels, test_features, test_labels)
        print ("global top3 NN accuracy = %f" % (nn_top3_acc*100))

    def _global_prob_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):
        #### sometimes directly using Logistic regression solver cannot give the best threshold, remember to try manual threshold

        reshaped_shadow_classes = np.reshape(shadow_classes,-1).astype(np.int64)
        total_shadow_num = len(reshaped_shadow_classes)
        reshaped_test_classes = np.reshape(test_classes,-1).astype(np.int64)
        total_test_num = len(reshaped_test_classes)

        train_features = np.reshape(shadow_confidence,(total_shadow_num,-1))
        train_features = [train_features[i,reshaped_shadow_classes[i]] for i in range(total_shadow_num)]
        train_features = np.reshape(train_features,(-1,1))

        test_features = np.reshape(test_confidence,(total_test_num,-1))
        test_features = [test_features[i,reshaped_test_classes[i]] for i in range(total_test_num)]
        test_features = np.reshape(test_features,(-1,1))

        train_features = np.nan_to_num(train_features)
        test_features = np.nan_to_num(test_features)
        train_labels = np.reshape(shadow_labels,(-1))
        test_labels = np.reshape(test_labels,(-1))

        model = LogisticRegression(random_state=0, solver='saga')

        model.fit(train_features, train_labels)
        print ("global class prob lr accuracy = %f " % (model.score(test_features, test_labels)*100))


    def _per_class_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):

        per_class_acc = 0.0
        nn_per_class_acc = 0.0

        reshaped_shadow_classes = np.reshape(shadow_classes,-1).astype(np.int64)
        total_shadow_num = len(reshaped_shadow_classes)
        reshaped_test_classes = np.reshape(test_classes,-1).astype(np.int64)
        total_test_num = len(reshaped_test_classes)
        shadow_confidence = np.reshape(shadow_confidence,(total_shadow_num,-1))
        test_confidence = np.reshape(test_confidence,(total_test_num,-1))
        shadow_labels = np.reshape(shadow_labels,-1)
        test_labels = np.reshape(test_labels,-1)

        #print (shadow_confidence.shape,reshaped_shadow_classes.shape,shadow_labels.shape)

        # per class test
        for i in range(self.num_classes):

            train_class_indices = np.arange(total_shadow_num)[reshaped_shadow_classes == i]
            test_class_indices = np.arange(total_test_num)[reshaped_test_classes == i]

            #print (i,train_class_indices,test_class_indices)

            this_train_features = shadow_confidence[train_class_indices]
            this_test_features = test_confidence[test_class_indices]
            this_train_labels = shadow_labels[train_class_indices]
            this_test_labels = test_labels[test_class_indices]

            this_train_features = np.nan_to_num(this_train_features)
            this_test_features = np.nan_to_num(this_test_features)

            ### LR per class test
            model = LogisticRegression(random_state=0, solver='lbfgs')
            model.fit(this_train_features, this_train_labels)
            lr_perclass_confidence_acc = model.score(this_test_features,this_test_labels)

            ### one layer NN per class test
            nn_perclass_confidence_acc = self._nn_attack(this_train_features, this_train_labels, this_test_features, this_test_labels)

            per_class_acc += lr_perclass_confidence_acc
            nn_per_class_acc += nn_perclass_confidence_acc

        print ("Class vector lr acc = %f" % (per_class_acc * 100 / self.num_classes))
        print ("Class vector nn acc = %f" % (nn_per_class_acc * 100 / self.num_classes))

    def _instance_distance_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):

        total_distance_acc = 0.0
        total_dist_precision = 0.0
        total_dist_recall = 0.0

        for i in range(self.membership_attack_number):

            this_train_features = shadow_confidence[:,i]
            this_train_labels = shadow_labels[:,i]
            this_test_features = test_confidence[:,i]
            this_test_labels = test_labels[:,i]
            this_class = shadow_classes[0,i]

            distance_acc,dist_precision,dist_recall = self._instance_attack_membership('distance',this_train_features,this_train_labels,this_test_features,this_test_labels, this_class)
            total_distance_acc += distance_acc
            total_dist_precision += dist_precision
            total_dist_recall += dist_recall

        print ('instance level KL-distance acc = %f' % (total_distance_acc * 100 / self.membership_attack_number))
        #print ("total distance precision = %f" % (total_dist_precision * 100 / self.membership_attack_number))
        #print ("total distance recall = %f" % (total_dist_recall * 100 / self.membership_attack_number))

    def _instance_prob_attack(self,shadow_confidence,shadow_classes,shadow_labels,test_confidence,test_classes,test_labels):

        total_ratio_acc = 0.0
        total_ratio_precision = 0.0
        total_ratio_recall = 0.0

        for i in range(self.membership_attack_number):

            this_train_features = shadow_confidence[:,i]
            this_train_labels = shadow_labels[:,i]
            this_test_features = test_confidence[:,i]
            this_test_labels = test_labels[:,i]
            this_class = shadow_classes[0,i]

            ratio_acc,ratio_precision,ratio_recall = self._instance_attack_membership('prob',this_train_features,this_train_labels,this_test_features,this_test_labels, this_class)

            total_ratio_acc += ratio_acc
            total_ratio_precision += ratio_precision
            total_ratio_recall += ratio_recall

        print ("instance level prob of correct label acc = %f" % (total_ratio_acc * 100 / self.membership_attack_number))
        #print ("total prob of correct label precision = %f" % (total_ratio_precision * 100 / self.membership_attack_number))
        #print ("total prob of correct label recall = %f" % (total_ratio_recall * 100 / self.membership_attack_number))


    def _instance_attack_membership(self,attack_name,shadow_confidence,shadow_labels,test_confidence,test_labels,this_class):

        ###
        shadow_labels = np.reshape(shadow_labels,-1)
        shadow_num = len(shadow_labels)
        shadow_confidence = np.reshape(shadow_confidence,(shadow_num,-1))
        test_labels = np.reshape(test_labels,-1)
        test_num = len(test_labels)
        test_confidence = np.reshape(test_confidence,(test_num,-1))

        ### show # of members and # of non-members in shadow examples
        total_in = np.count_nonzero(shadow_labels)
        total_out = len(shadow_labels) - total_in
        if (total_in == 0 or total_out == 0):
            return [0.5] * 3


        if (attack_name == 'distance'):
            train_in_indices = []
            train_out_indices = []
            for i in range(len(shadow_labels)):
                if (shadow_labels[i] == 1):
                    train_in_indices.append(i)
                else:
                    train_out_indices.append(i)
            train_in_conf = shadow_confidence[np.array(train_in_indices)]
            train_out_conf = shadow_confidence[np.array(train_out_indices)]

            in_avg_conf = np.average(train_in_conf, axis=0)
            out_avg_conf = np.average(train_out_conf, axis=0)

            corr = 0
            import scipy
            from scipy.stats import entropy
            y_pred = np.zeros((len(test_labels)))
            for i in range(len(test_labels)):
                in_distance = scipy.stats.entropy(in_avg_conf, test_confidence[i])
                out_distance = scipy.stats.entropy(out_avg_conf, test_confidence[i])
                y_pred[i] = (in_distance < out_distance)
                if (in_distance < out_distance and test_labels[i] == 1):
                    corr += 1
                if (out_distance < in_distance and test_labels[i] == 0):
                    corr += 1
            distance_acc = corr * 1.0 / len(test_labels)
            dist_precision = precision_score(test_labels, y_pred, average='macro')
            dist_recall = recall_score(test_labels, y_pred, average='macro')

            return distance_acc,dist_precision,dist_recall

        if (attack_name == 'prob'):
            corr_class = int(this_class)

            train_features = shadow_confidence[:, corr_class]
            train_features = np.log(0.01 + train_features) - np.log(1.01 - train_features)
            train_features = np.reshape(train_features, (shadow_num, -1))

            test_features = test_confidence[:, corr_class]
            test_features = np.log(0.01 + test_features) - np.log(1.01 - test_features)
            test_features = np.reshape(test_features, (test_num, -1))

            train_labels = np.reshape(shadow_labels,-1)
            test_labels = np.reshape(test_labels,-1)

            model = LogisticRegression(random_state=0, solver='saga')
            model.fit(train_features, train_labels)
            ratio_confidence_acc = model.score(test_features, test_labels)
            precision = precision_score(test_labels, model.predict(test_features), average='macro')
            recall = recall_score(test_labels, model.predict(test_features), average='macro')

            return ratio_confidence_acc,precision,recall

        pass

    def _nn_attack(self,train, train_label, test, test_label):

        dem = train.shape[1]
        total_num = train.shape[0]
        test_total_num = test.shape[0]

        train = np.reshape(train, (total_num, 1, 1, dem))
        test = np.reshape(test, (test_total_num, 1, 1, dem))

        train = part_pytorch_dataset(train, train_label, train=True, transform=transforms.ToTensor())
        test = part_pytorch_dataset(test, test_label, train=False, transform=transforms.ToTensor())

        epochs = 100

        attack_model = onelayer_AttackNet(dem=dem) ## check the model details in model dir

        dtype = torch.FloatTensor
        label_type = torch.LongTensor
        criterion = nn.CrossEntropyLoss()

        attack_model.type(dtype)

        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=1e-7)

        train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)

        for epoch in range(epochs):
            for images, labels in train_loader:
                images = Variable(images).type(dtype)
                labels = Variable(labels).type(label_type)
                optimizer.zero_grad()
                outputs = attack_model(images)
                loss = criterion(outputs, labels)
                total_loss = loss
                total_loss.backward()
                optimizer.step()

        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images).type(dtype)
            outputs = attack_model(images)
            labels = labels.type(label_type)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        acc = correct.item() * 1.0
        acc = acc / total
        testing_acc = acc

        return testing_acc

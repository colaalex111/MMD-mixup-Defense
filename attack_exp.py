
from blackbox_attack import *
import argparse
from data import dataset
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules

def choose_model(target_dataset,model_name):
    if (target_dataset.dataset_name == 'cifar100'):
        if (model_name == 'alexnet'):
            target_model = alexnet(num_classes=100)
        if (model_name == 'resnet20'):
            target_model = resnet(depth=20, num_classes=100)
            # target_model = convert_batchnorm_modules(resnet(depth=20, num_classes=100),
            #                                         converter=_batchnorm_to_groupnorm_new)

        if (model_name == 'resnet110'):
            target_model = resnet(depth=110, num_classes=100)
        if (model_name == 'densenet_cifar'):
            target_model = densenet(depth=100, num_classes=100)
            # target_model = convert_batchnorm_modules(densenet(depth=100, num_classes=100),
            #                                         converter=_batchnorm_to_groupnorm_new)
        if (model_name == 'resnet101'):
            model = models.resnet101(pretrained=args.pretrained)
            number_features = model.fc.in_features
            model.fc = nn.Linear(number_features, 100)
            model.avgpool = nn.AvgPool2d(1, 1)
            target_model = model
        if (model_name == 'resnet18'):
            model = models.resnet18(pretrained=args.pretrained)
            number_features = model.fc.in_features
            model.fc = nn.Linear(number_features, 100)
            model.avgpool = nn.AvgPool2d(1, 1)
            target_model = model
        if (model_name == 'vgg16'):
            target_model = models.vgg16(pretrained=args.pretrained)
            target_model.avgpool = nn.Sequential(nn.Flatten())
            target_model.classifier = nn.Sequential(nn.Linear(512, 100))
            # target_model.classifier = nn.Sequential(nn.Conv2d(512,100,kernel_size=1),
            #                                        nn.AvgPool2d(kernel_size=1))

        if (model_name == 'densenet121'):
            target_model = models.densenet121(pretrained=args.pretrained)
            target_model.classifier = nn.Linear(1024, 100)

    elif (target_dataset.dataset_name  == 'cifar10'):
        if (model_name == 'alexnet'):
            target_model = alexnet(num_classes=10)
        if (model_name == 'resnet20'):
            target_model = resnet(depth=20, num_classes=10)
            # target_model = convert_batchnorm_modules(resnet(depth=20, num_classes=10),
            #                                         converter=_batchnorm_to_groupnorm_new)
        if (model_name == 'resnet110'):
            target_model = resnet(depth=110, num_classes=10)
        if (model_name == 'densenet_cifar'):
            target_model = densenet(depth=100, num_classes=10)
            # target_model = convert_batchnorm_modules(densenet(depth=100, num_classes=10),
            #                                         converter=_batchnorm_to_groupnorm_new)
        if (model_name == 'resnet101'):
            model = models.resnet101(pretrained=args.pretrained)
            number_features = model.fc.in_features
            model.fc = nn.Linear(number_features, 10)
            model.avgpool = nn.AvgPool2d(1, 1)
            target_model = model
        if (model_name == 'resnet18'):
            model = models.resnet18(pretrained=args.pretrained)
            number_features = model.fc.in_features
            model.fc = nn.Linear(number_features, 10)
            model.avgpool = nn.AvgPool2d(1, 1)
            target_model = model
        if (model_name == 'vgg16'):
            target_model = models.vgg16(pretrained=args.pretrained)
            target_model.avgpool = nn.Sequential(nn.Flatten())
            target_model.classifier = nn.Sequential(nn.Linear(512, 10))
        if (model_name == 'densenet121'):
            target_model = models.densenet121(pretrained=args.pretrained)
            target_model.classifier = nn.Sequential(nn.Linear(1024, 10))
    else:
        target_model = TargetNet(args.dataset, target_dataset.data.shape[1], len(np.unique(target_dataset.label)))

    return target_model

def _batchnorm_to_groupnorm_new(module):
    return nn.GroupNorm(num_groups=module.num_features, num_channels=module.num_features, affine=True)


def eval_model(target_model, train, test, eval_attack,validation, learning_rate, decay, epochs,
               batch_size, starting_index):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model.to(device)

    if (args.dataset == 'cifar10'):
        num_classes = 10
    elif (args.dataset == 'cifar100'):
        num_classes = 100
    elif (args.dataset == 'purchase'):
        num_classes = 100
    elif (args.dataset == 'texas'):
        num_classes = 100
    elif (args.dataset == 'mnist'):
        num_classes = 10

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate, weight_decay=decay, momentum=0.9)
    if (args.dataset=='purchase' or args.dataset=='texas'):
        optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=decay)

    if (args.dp_sgd):
        ### adding dp components
        privacy_engine = PrivacyEngine(
            target_model,
            batch_size,
            args.target_data_size,  ### overall training set size
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),  ### params for renyi dp
            noise_multiplier=args.noise_scale,  ### sigma
            max_grad_norm=args.grad_norm,  ### this is from dp-sgd paper
        )
        privacy_engine.attach(optimizer)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=False,num_workers=1)
    train_loader_in_order = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=1)

    ## some info we need to record for MI attacks on validation set
    validation_confidence_in_training = []
    validation_label_in_training = []
    avg_loss = 0

    for epoch in range(epochs):
        avg_loss = 0
        target_model.train()

        if (epoch in args.schedule):
            learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print ("new learning rate = %f" % (learning_rate))

        for index,(images, labels) in enumerate(train_loader):

            if (args.mixup):
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, args.alpha)
                optimizer.zero_grad()
                inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
                outputs = target_model(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, outputs)
                avg_loss+=loss.item()
                loss.backward()
                optimizer.step()

            else:
                # normal training
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()

        avg_loss = avg_loss/(index+1)

        ### test train/valid acc after every epoch
        correct = 0.0
        total = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            outputs = target_model(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0
        this_training_acc = acc

        print ('Training Accuracy %f' %(acc))

        correct = 0.0
        total = 0.0
        target_model.eval()
        for images, labels in validation_loader:
            images = images.to(device)
            outputs = target_model(images)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc = correct.item()
        acc = acc / total
        acc = acc * 100.0

        print('Validation Accuracy %f ' % (acc))

        this_validation_acc = acc
        ### if the gap is less than 3%, then don't do MMD regularization. This threshold can be tuned.
        ### if this MMD loss regularization is applied in every epoch, this will result in unstable training.
        if (abs(this_validation_acc - this_training_acc) < 3 or args.mmd_loss_lambda<1e-5):
            continue
        else:
            pass

        validation_label_in_training = []
        validation_confidence_in_training = []
        for loss_index,(train_images,train_labels) in enumerate(train_loader_in_order):

            batch_num = train_labels.size()[0]
            optimizer.zero_grad()

            ### create a validation batch so that the number of instances in each class is the same as the training batch
            valid_images = torch.zeros_like(train_images).type(torch.FloatTensor).to(device)
            valid_labels = torch.zeros_like(train_labels).type(torch.LongTensor).to(device)
            valid_index = 0
            for label_index,i in enumerate(torch.unique(train_labels)):
                this_frequency = torch.bincount(train_labels)[i].to(device)
                this_class_start = starting_index[i]  ## i is the current class label

                if (i<num_classes-1):
                    this_class_end = starting_index[i+1]-1
                else:
                    this_class_end = validation.__len__()-1

                for j in range(this_frequency):
                    random_index = np.random.randint(this_class_start,this_class_end)
                    new_images,new_labels =((validation).__getitem__(random_index))
                    valid_images[valid_index] = new_images.to(device)
                    valid_labels[valid_index] = (torch.ones(1)*new_labels).type(torch.LongTensor).to(device)
                    valid_index+=1

            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            outputs = target_model(train_images)
            all_train_outputs = F.softmax(outputs,dim=1)
            #all_train_outputs = all_train_outputs.view(-1,num_classes)
            train_labels = train_labels.view(batch_num,1)

            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)
            outputs = target_model(valid_images)
            all_valid_outputs = F.softmax(outputs,dim=1)
            all_valid_outputs = (all_valid_outputs).detach_()
            valid_labels = valid_labels.view(batch_num,1)

            validation_label_in_training.append(valid_labels.cpu().data.numpy()) ### this is to get the data for MI attacks on Validation set
            validation_confidence_in_training.append(all_valid_outputs.cpu().data.numpy())

            if (args.mmd_loss_lambda>0):
                mmd_loss = mix_rbf_mmd2(all_train_outputs,all_valid_outputs,sigma_list=[1])*args.mmd_loss_lambda
                mmd_loss.backward()

            ### MMD regularization shouldn't be applied in the last training epoch for better testing accuracy
            if (epoch != epochs-1):
                optimizer.step()

    print ("TRAINING FINISHED")

    ### stats for dp-sgd
    if (args.dp_sgd):
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print ("eps:", epsilon)
        print ("best alphas:", best_alpha)

    ### train/validation accuracy after training
    correct = 0.0
    total = 0.0
    target_model.eval()

    for images, labels in train_loader:
        images = images.to(device)
        outputs = target_model(images)
        labels = labels.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc = correct.item()
    acc = acc / total
    acc = acc * 100.0
    print("Train Accuracy %f " % (acc))
    training_acc = acc

    correct = 0.0
    total = 0.0

    testing_confidence = []
    testing_label = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = target_model(images)
        labels = labels.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        for i in range(images.size()[0]):
            testing_confidence.append(F.softmax(outputs[i],dim=0).detach().cpu().data.numpy())
        testing_label.append(labels.detach().cpu().data.numpy())

    acc = correct.item()
    acc = acc / total
    acc = acc * 100.0
    print("Test Accuracy %f " % (acc))
    testing_acc = acc

    ### generate predictions for membership evaluation set

    eval_loader = torch.utils.data.DataLoader(eval_attack, batch_size=100, shuffle=False,num_workers=1)
    confidence = []
    for index, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        labels = labels.to(device)
        this_confidence = target_model(images)
        for i in range(images.size()[0]):
            confidence.append(F.softmax(this_confidence[i],dim=0).cpu().detach().data.numpy())
    confidence = np.array(confidence)
    confidence = np.reshape(confidence, (args.membership_attack_number, num_classes))

    return target_model,confidence,training_acc,testing_acc,validation_confidence_in_training,validation_label_in_training,avg_loss,testing_confidence,testing_label


def attack_experiment():

    ### dataset && membership inference data
    membership_attack_number = args.membership_attack_number
    target_dataset = dataset(dataset_name=args.dataset, data_path=args.data_path,
                             membership_attack_number=membership_attack_number)
    num_classes = len(np.unique(target_dataset.label))

    shadow_classes = np.zeros((args.shadow_model_number, membership_attack_number))
    shadow_confidence = np.zeros((args.shadow_model_number, membership_attack_number,
                                 len(np.unique(target_dataset.train_label))))
    shadow_label = np.zeros((args.shadow_model_number, membership_attack_number))
    shadow_avg_loss = np.zeros((args.shadow_model_number))
    shadow_valid_confidence = []
    shadow_valid_classes = []
    shadow_test_confidence = []
    shadow_test_classes = []
    shadow_training_acc = 0
    shadow_testing_acc = 0
    shadow_training_acc_list = []
    shadow_testing_acc_list = []

    ## get info for shadow models
    for i in range(args.shadow_model_number):
        print ("shadow model number %d" %(i))
        ### generate training / testing / validation data
        target_data_number = args.target_data_size
        train, test, eval_attack,validation, eval_partition, in_train_partition, out_train_partition,starting_index,train_eval = target_dataset.select_part(
            target_data_number, membership_attack_number,shadow_model_label=1)

        ### membership information for eval set
        for j in range(len(in_train_partition)):
            shadow_label[i, in_train_partition[j]] = 1
        for j in range(len(out_train_partition)):
            shadow_label[i, out_train_partition[j]] = 0

        target_model = choose_model(target_dataset,args.model_name)

        ###train shadow models & get confidence for train/test data
        target_model,confidence,training_acc,testing_acc,this_valid_conf,this_valid_classes,this_avg_loss,this_test_conf,this_test_classes = \
            eval_model(target_model, train, test, eval_attack,validation,
                                learning_rate=args.target_learning_rate,
                       decay=args.target_l2_ratio, epochs=args.target_epochs,
                                batch_size=args.target_batch_size,
                                starting_index=starting_index)

        ### record everything we need
        shadow_classes[i] = np.copy(target_dataset.part_eval_label)
        shadow_confidence[i] = confidence
        shadow_avg_loss[i] = this_avg_loss
        shadow_training_acc+=training_acc
        shadow_testing_acc+=testing_acc
        shadow_training_acc_list.append(training_acc)
        shadow_testing_acc_list.append(testing_acc)

        ### for validation MI
        shadow_valid_confidence.append(this_valid_conf)
        shadow_valid_classes.append(this_valid_classes)
        shadow_test_confidence.append(this_test_conf)
        shadow_test_classes.append(this_test_classes)

    print ("shadow model, avg training acc = %f"%(shadow_training_acc/args.shadow_model_number))
    print ("shadow model, avg testing acc = %f"%(shadow_testing_acc/args.shadow_model_number))


    ### get info for testing models
    test_classes = np.zeros((args.test_model_number, membership_attack_number))
    test_confidence = np.zeros((args.test_model_number, membership_attack_number,
                                 len(np.unique(target_dataset.train_label))))
    test_label = np.zeros((args.test_model_number, membership_attack_number))
    test_avg_loss = np.zeros((args.test_model_number))

    test_training_acc = 0
    test_testing_acc = 0
    test_training_acc_list = []
    test_testing_acc_list = []

    test_valid_confidence = []
    test_valid_classes = []
    test_test_confidence = []
    test_test_classes = []

    ## get info for test models
    for i in range(args.test_model_number):
        print ("test model number %d" %(i))
        ### generate training / testing / validation data
        target_data_number = args.target_data_size
        train, test, eval_attack,validation, eval_partition, in_train_partition, out_train_partition,starting_index,train_eval = target_dataset.select_part(
            target_data_number, membership_attack_number,shadow_model_label=0)

        ### membership information for eval set
        for j in range(len(in_train_partition)):
            test_label[i, in_train_partition[j]] = 1
        for j in range(len(out_train_partition)):
            test_label[i, out_train_partition[j]] = 0

        target_model = choose_model(target_dataset,args.model_name)

        ###train shadow models & get confidence for train/test data
        target_model,confidence,training_acc,testing_acc,this_valid_conf,this_valid_classes,this_avg_loss,this_test_conf,this_test_classes = \
            eval_model(target_model, train, test, eval_attack,validation,
                                learning_rate=args.target_learning_rate,
                       decay=args.target_l2_ratio, epochs=args.target_epochs,
                                batch_size=args.target_batch_size,
                                starting_index=starting_index)

        ### record everything we need
        test_classes[i] = np.copy(target_dataset.part_eval_label)
        test_confidence[i] = confidence
        test_avg_loss[i] = this_avg_loss
        test_training_acc+=training_acc
        test_testing_acc+=testing_acc
        test_training_acc_list.append(training_acc)
        test_testing_acc_list.append(testing_acc)

        ### for validation MI
        test_valid_confidence.append(this_valid_conf)
        test_valid_classes.append(this_valid_classes)
        test_test_confidence.append(this_test_conf)
        test_test_classes.append(this_test_classes)


    print ("test model, avg training acc = %f"%(test_training_acc/args.test_model_number))
    print ("test model, avg testing acc = %f"%(test_testing_acc/args.test_model_number))

    #### launch blackbox attacks
    ## baseline attack
    attack = blackbox_attack(args.membership_attack_number,name='baseline',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label)
    ## avgloss attack
    attack = blackbox_attack(args.membership_attack_number,name='avg_loss',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label,avg_loss=np.average(shadow_avg_loss))
    ## top1 attack
    attack = blackbox_attack(args.membership_attack_number,name='top1',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label)
    ## top3 attack
    attack = blackbox_attack(args.membership_attack_number,name='top3',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label)
    ## global prob attack
    attack = blackbox_attack(args.membership_attack_number,name='global_prob',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label)
    ## per class attack
    attack = blackbox_attack(args.membership_attack_number,name='per_class',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label)
    ## instance distance attack
    attack = blackbox_attack(args.membership_attack_number,name='instance_distance',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label)
    ## instance prob/ratio attack
    attack = blackbox_attack(args.membership_attack_number,name='instance_prob',num_classes=num_classes)
    attack.attack(shadow_confidence,shadow_classes,shadow_label,test_confidence,test_classes,test_label)


    #### validation set MI attacks

    if (args.validation_mi):

        print ("----validation MI test----")

        shadow_valid_confidence = np.array(shadow_valid_confidence)
        shadow_valid_confidence = np.reshape(shadow_valid_confidence,(-1,num_classes))
        shadow_valid_classes = np.array(shadow_valid_classes)
        shadow_valid_classes = np.reshape(shadow_valid_classes,(-1))
        shadow_test_confidence = np.array(shadow_test_confidence)
        shadow_test_confidence = np.reshape(shadow_test_confidence,(-1,num_classes))
        shadow_test_classes = np.array(shadow_test_classes)
        shadow_test_classes = np.reshape(shadow_test_classes,(-1))

        test_valid_confidence = np.array(test_valid_confidence)
        test_valid_confidence = np.reshape(test_valid_confidence,(-1,num_classes))
        test_valid_classes = np.array(test_valid_classes)
        test_valid_classes = np.reshape(test_valid_classes,(-1))
        test_test_confidence = np.array(test_test_confidence)
        test_test_confidence = np.reshape(test_test_confidence,(-1,num_classes))
        test_test_classes = np.array(test_test_classes)
        test_test_classes = np.reshape(test_test_classes,(-1))

        min_len = min(len(shadow_test_classes),len(shadow_valid_classes))
        #print ("min len",min_len)

        shadow_confidence = np.concatenate((shadow_valid_confidence[:min_len,:],shadow_test_confidence[:min_len,:]))
        shadow_classes = np.concatenate((shadow_valid_classes[:min_len],shadow_test_classes[:min_len]))
        shadow_labels = np.concatenate((np.ones_like(shadow_valid_classes[:min_len]),np.zeros_like(shadow_test_classes[:min_len])))
        shadow_labels = np.reshape(shadow_labels,(-1))

        min_len = min(len(test_test_classes),len(test_valid_classes))
        #print ("min len",min_len)

        test_confidence = np.concatenate((test_valid_confidence[:min_len,:],test_test_confidence[:min_len,:]))
        test_classes = np.concatenate((test_valid_classes[:min_len],test_test_classes[:min_len]))
        test_labels = np.concatenate((np.ones_like(test_valid_classes[:min_len]),np.zeros_like(test_test_classes[:min_len])))
        test_labels = np.reshape(test_labels,(-1))

        shadow_confidence = np.nan_to_num(shadow_confidence)
        test_confidence = np.nan_to_num(test_confidence)

        #print (shadow_confidence.shape,shadow_classes.shape,shadow_labels.shape)
        #print (test_confidence.shape,test_classes.shape,test_labels.shape)

        ## baseline attack
        attack = blackbox_attack(args.membership_attack_number, name='baseline', num_classes=num_classes)
        attack.attack(shadow_confidence, shadow_classes, shadow_labels, test_confidence, test_classes, test_labels)
        ## top1 attack
        attack = blackbox_attack(args.membership_attack_number, name='top1', num_classes=num_classes)
        attack.attack(shadow_confidence, shadow_classes, shadow_labels, test_confidence, test_classes, test_labels)
        ## top3 attack
        attack = blackbox_attack(args.membership_attack_number, name='top3', num_classes=num_classes)
        attack.attack(shadow_confidence, shadow_classes, shadow_labels, test_confidence, test_classes, test_labels)
        ## global prob attack
        attack = blackbox_attack(args.membership_attack_number, name='global_prob', num_classes=num_classes)
        attack.attack(shadow_confidence, shadow_classes, shadow_labels, test_confidence, test_classes, test_labels)
        ## per class attack
        attack = blackbox_attack(args.membership_attack_number, name='per_class', num_classes=num_classes)
        attack.attack(shadow_confidence, shadow_classes, shadow_labels, test_confidence, test_classes, test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',type=str,default='/home/lijiacheng/dataset/') ### data file path
    parser.add_argument('--target_data_size', type=int, default=3000) ### training set size for target model
    parser.add_argument('--model_name',type=str,default='alexnet') ### model name
    parser.add_argument('--target_learning_rate', type=float, default=0.01) ### target model learning rate
    parser.add_argument('--attack_learning_rate', type=float, default=0.001)### attack model learning rate (if needed)
    parser.add_argument('--target_batch_size', type=int, default=100) ### target model batch size
    parser.add_argument('--attack_batch_size', type=int, default=100) ### attack model batch size
    parser.add_argument('--target_epochs', type=int, default=20) ### target model training epochs
    parser.add_argument('--attack_epochs', type=int, default=500) ### attack model trainingpeochs (if needed)
    parser.add_argument('--target_l2_ratio', type=float, default=5e-4) ### l2 regularizer weight
    parser.add_argument('--dataset', type=str, default='mnist') ### dataset name
    parser.add_argument('--shadow_model_number', type=int, default=10) ### total number of shadow models
    parser.add_argument('--test_model_number',type=int,default=10) ### total number of testing models
    parser.add_argument('--schedule', type=int, nargs='+', default=[80,120]) ### target model LR schedule
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--membership_attack_number', type=int, default=500) ### size of the membership evaluation set
    parser.add_argument('--pretrained',type=int,default=0) ###  pretrained model from torchvision
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--alpha',type=float,default='1.0') ### param for mixup
    parser.add_argument('--mixup',type=int,default=0) ### mixup

    parser.add_argument('--mmd_loss_lambda',type=float,default=0) ### weight for MMD loss

    parser.add_argument('--dp_sgd',type=int,default=0) ### if 1 then dp_sgd is applied
    parser.add_argument('--delta',type=float,default=1e-5)
    parser.add_argument('--grad_norm',type=float,default=10000)
    parser.add_argument('--noise_scale',type=float,default=0.0001)

    parser.add_argument('--validation_mi',type=int,default=0) ### running MI attacks on Validation set if this is set to 1

    ### some settings
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_printoptions(threshold=5000, edgeitems=20)
    torch.manual_seed(123)
    np.random.seed(123)

    args = parser.parse_args()
    print (vars(args))

    attack_experiment()

    print (vars(args))


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import time
import os
from tensorboardX import SummaryWriter
from RNN import RNN
from data import get_data
from data_test import get_data_test
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# GPU
if torch.cuda.is_available():
    print('GPU Device Name:', torch.cuda.get_device_name(0))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("{} is Applied".format(str(device)))

###############################


class TENG:
    def __init__(self):
        ############visualize################

        self.log_dir = './tf'
        self.writer = SummaryWriter(self.log_dir)

        self.input_size = 9
        self.output_size = 25
        self.hidden_size = 300
        self.num_layers = 5
        self.learning_rate = 0.01  # 0.1
        self.sequence_length = 1

        self.batch_size = 100  # 400
        self.epochs = 50

        self.model = RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                         num_layers=self.num_layers, output_size=self.output_size)
        # print('\nModel Info: ', self.model)
        '''
        (rnn): RNN(9, 25, num_layers=2, batch_first=True, dropout=0.1)
        (fc): Linear(in_features=25, out_features=25, bias=True)
        (relu): ReLU()
        '''
        print(self.model.rnn)
        self.model.to(device)
        self.loss_function = nn.CrossEntropyLoss()
        # self.updateParams = optim.SGD(
        # self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.updateParams = optim.Adam(
            self.model.parameters(), weight_decay=5e-4, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.updateParams, milestones=[10, 20, 30], gamma=0.1)

        #######get the data########
        train_datasets, test_datasets = get_data()
        self.train_loader = torch.utils.data.DataLoader(
            train_datasets, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            test_datasets, batch_size=self.batch_size, shuffle=True)

        # mini-batch
        print('#####Model Initialization is completed and ready for the training process.#####')
        print('\n')
        time.sleep(0.1)
        model_file = "better_RNN_model_checkpoint.pth.tar"
        if os.path.isfile(model_file):
            print("#############Loading the pre-trained model#############")
            checkpoint = torch.load(model_file)
            self.start_epoch = checkpoint['epoch']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.updateParams.load_state_dict(checkpoint['optimizer'])
            self.training_accuracy = checkpoint['training_accuracy']
            self.validation_accuracy = checkpoint['validation_accuracy']
            self.training_loss = checkpoint['training_loss']
            self.validation_loss = checkpoint['validation_loss']
            self.time_list = checkpoint['time']
            print('\n')
            print('preivous model accuracy:', self.best_accuracy)
            print('\n')
        else:
            self.start_epoch = 0
            self.best_accuracy = 0
            self.training_accuracy = []
            self.validation_accuracy = []
            self.training_loss = []
            self.validation_loss = []
            self.time_list = []
            print('NEW model accuracy:', self.best_accuracy)

    def train(self):

        def save_checkpoint(state, better, file='RNN_model_checkpoint.pth.tar'):
            torch.save(state, file)
            if better:
                shutil.copyfile(
                    file, 'better_RNN_model_checkpoint.pth.tar')

        def training(epochs):

            step = 0
            self.model.train()  # initializing the training
            print("CNN training starts__epoch: {}, LR= {}".format(
                epochs, self.scheduler.get_lr()))
            training_loss = 0
            total = 0
            final_score = 0
            self.scheduler.step()  # dynamically change the learning rate
            self.loss = 0

            for batch_id, (X_batch, y_batch) in enumerate(self.train_loader):

                X_batch = X_batch.view(-1,
                                       self.sequence_length, self.input_size)

                X_batch = X_batch.float().to(device)
                y_batch = y_batch.to(device)
                y_batch = y_batch.to(device).detach()
                if X_batch.requires_grad:
                    pass
                else:
                    print('AutoGrad is OFF!')

                self.updateParams.zero_grad()  # zero gradient before the backward
                result = self.model(X_batch)
                batch_loss = self.loss_function(result, y_batch)
                # wihout .item(),in gpu model, not enough memory
                training_loss += batch_loss.item()
                batch_loss.backward()
                self.updateParams.step()  # performs a parameter update based on the current gradient
                _, predict = torch.max((result), 1)  # dim=1->each row
                final_score += predict.eq(y_batch).cpu(
                ).sum().type(torch.DoubleTensor).item()

                # check the gradient
                # print('ID', batch_id)
                # print('after back prop--parameter: ', list(self.model.parameters())
                #       [0].grad)  # the gradient is so very small

            training_loss_mean = training_loss / \
                (len(self.train_loader.dataset)/(self.batch_size))
            training_accuracy = 100*final_score / \
                (len(self.train_loader.dataset))
            print(
                "Training-epoch-{}-training_loss_mean: {:.4f}".format(epochs, training_loss_mean))
            print(
                "Training-epoch-{}-training_accuracy: {:.4f}%".format(epochs, training_accuracy))
            # self.writer.add_image('Output', vutils.make_grid(output.data, normalize=True, scale_each=True), niter)
            return (training_loss_mean, training_accuracy)

        def validation(epochs):

            self.model.eval()
            validation_loss = 0
            total = 0
            final_score = 0

            with torch.no_grad():  # temporarily set all the requires_grad flag to false

                for batch_id, (test_data, target_test) in enumerate(self.test_loader):

                    test_data = test_data.view(-1,
                                               self.sequence_length, self.input_size)

                    test_data = test_data.float().to(device)
                    target_test = target_test.to(device)

                    result = self.model(test_data)
                    batch_loss = self.loss_function(result, target_test)
                    validation_loss += batch_loss
                    _, predict = torch.max(
                        (result), 1)  # dim=1->each row
                    final_score += predict.eq(target_test).cpu(
                    ).sum().type(torch.DoubleTensor).item()

            validation_loss_mean = validation_loss / \
                (len(self.test_loader.dataset)/(self.batch_size))
            validation_accuracy = 100*final_score / \
                (len(self.test_loader.dataset))

            print(
                "Validation-epoch-{}-Validation_loss_mean: {:.4f}".format(epochs, validation_loss_mean))
            print('Validation Accuracy: {:.4f}%'.format(validation_accuracy))

            self.model_accuracy_cur_epoch = validation_accuracy

            return (validation_loss_mean, validation_accuracy)

        if __name__ == "__main__":
            print("######CIFAR100 Training-Validation Starts######")
            epoch_iter = range(1, self.epochs)

            self.model_accuracy_cur_epoch = 0
            if self.start_epoch == self.epochs:
                pass
            else:
                for i in range(self.start_epoch+1, self.epochs):
                    time_begin = time.time()
                    training_result = training(i)
                    self.training_loss.append(training_result[0])
                    self.training_accuracy.append(training_result[1])
                    vali_result = validation(i)
                    self.validation_loss.append(vali_result[0])
                    self.validation_accuracy.append(vali_result[1])
                    time_end = time.time()-time_begin
                    self.time_list.append(time_end)
                    progress = float(i*100//len(epoch_iter))
                    print('Progress: {:.4f}%'.format(progress))
                    print('\n')
                    #######################################
                    # Tensorboard Visualization
                    niter = i
                    # tensorboard --logdir=tf --port 6066

                    self.writer.add_scalars(
                        'Loss Curve',
                        {'Training Loss': training_result[0],
                         'Validation Loss': vali_result[0]}, niter
                    )  # attention->add_scalarS

                    self.writer.add_scalars(
                        'Accuracy Curve',
                        {'Training Accuracy': training_result[1],
                         'Validation Accuracy': vali_result[1]}, niter
                    )

                   #######################################
                    better = self.model_accuracy_cur_epoch > self.best_accuracy
                    self.best_accuracy = max(
                        self.best_accuracy, self.model_accuracy_cur_epoch)
                    # if better:
                    #    torch.save(self.model.state_dict(), 'CNN_MODEL.pt')
                    save_checkpoint({'epoch': i,
                                     'best_accuracy': self.best_accuracy,
                                     'state_dict': self.model.state_dict(),
                                     'optimizer': self.updateParams.state_dict(),
                                     'training_loss': self.training_loss,
                                     'training_accuracy': self.training_accuracy,
                                     'validation_loss': self.validation_loss,
                                     'validation_accuracy': self.validation_accuracy,
                                     'time': self.time_list,
                                     }, better-1)
                    print('Model Updated, proceeding to next epoch, best accuracy= {}'.format(
                        self.best_accuracy))
                # save the model after training
                torch.save(self.model.state_dict(), 'CNN_MODEL.pt')

            # ploting

            # loss function

            plt.figure(1)
            sns.set_style('whitegrid')
            plt.plot(epoch_iter, self.training_loss, color='red', linestyle='solid', linewidth='3.0',
                     marker='p', markerfacecolor='red', markersize='10', label='Training Loss')
            plt.plot(epoch_iter, self.validation_loss, color='green', linestyle='solid', linewidth='3.0',
                     marker='o', markerfacecolor='green', markersize='10', label='Validation Loss')
            plt.ylabel('Loss', fontsize=18)
            plt.xlabel('Epochs', fontsize=18)
            title = "RNN Result-loss"
            plt.title(title, fontsize=12)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.show()

            # Training accuracy
            plt.figure(2)
            sns.set_style('whitegrid')
            plt.plot(epoch_iter, self.training_accuracy, color='blue', linestyle='solid', linewidth='3.0',
                     marker='s', markerfacecolor='blue', markersize='10', label='Training Accuracy')
            plt.plot(epoch_iter, self.validation_accuracy, color='green', linestyle='solid', linewidth='3.0',
                     marker='s', markerfacecolor='green', markersize='10', label='Validation Accuracy')
            title = "RNN Result-accuracy"
            plt.title(title, fontsize=12)
            plt.xlabel('Epochs', fontsize=18)
            plt.title("Model Accuracy", fontsize=14)
            plt.legend(fontsize=14)
            plt.show()

            plt.figure(3)
            sns.set_style('whitegrid')
            plt.plot(epoch_iter, self.time_list, color='blue', linestyle='solid', linewidth='3.0',
                     marker='s', markerfacecolor='blue', markersize='10', label='Validation Loss')
            plt.ylabel('Time (s)', fontsize=18)
            plt.xlabel('Epochs', fontsize=18)
            plt.title("Speed", fontsize=14)
            plt.legend(fontsize=14)
            plt.show()

    def forward_EM(self, filepath, target):
        # data processing
        df = pd.read_csv(filepath, header=None)  # no column names!!!

        df_x = df.iloc[:, :9]
        df_x = df_x.div(df_x.sum(axis=1), axis=0)  # normalize

        X = df_x
        X_scaling = StandardScaler().fit_transform(X)  # numpy.array
        input_data = torch.tensor(X_scaling, requires_grad=True)
        input_data = input_data.view(-1, self.sequence_length, self.input_size)

        y_new = df.iloc[:, -1]
        y_new -= 1

        input_data = input_data.float().to(device)

        ##############
        self.model.eval()
        result = self.model(input_data)
        _, predict = torch.max(result, 1)
        predict = predict.cpu()
        i = 0
        for elem in predict:
            if elem == target:
                i += 1
        # for i in range(len(predict)):
        #    # print(predict)
        #    if predict[i] == y_new[i]:
        #        count += 1
        acc = float(i/len(predict))
        # print('Accuracy: {}%'.format(acc*100))
        # from sklearn.metrics import confusion_matrix

        # confusion_matrix = confusion_matrix(
        #     y_true=y_new, y_pred=predict)

        # # #Normalize CM
        # confusion_matrix = cm = confusion_matrix.astype(
        #     'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        # df_cm = pd.DataFrame(confusion_matrix)

        # # plot confusion matrix
        # fig, ax = plt.subplots()
        # sns.heatmap(df_cm, cmap="coolwarm", annot=False)
        # fig.set_size_inches(8, 6)
        # ax.set_title("Confusion Matrix of RNN, Data: {}".format(filepath))
        # ax.set_xlabel('Perdicted Label', fontsize=12)
        # ax.set_ylabel('Actual Label', fontsize=12)

        # plt.show()

        return predict, acc

    def forward_ni(self, filepath):
        # data processing
        df = pd.read_csv(filepath, header=None)  # no column names!!!

        df_x = df.iloc[:, :9]
        df_x = df_x.div(df_x.sum(axis=1), axis=0)  # normalize

        X = df_x
        X_scaling = StandardScaler().fit_transform(X)  # numpy.array
        input_data = torch.tensor(X_scaling, requires_grad=True)
        input_data = input_data.view(-1, self.sequence_length, self.input_size)

        y_new = df.iloc[:, -1]

        input_data = input_data.float().to(device)

        ##############
        self.model.eval()
        result = self.model(input_data)
        _, predict = torch.max(result, 1)
        predict = predict.cpu()
        predict = predict.numpy()
        i = 0
        print(predict)
        print(y_new.head(10))
        count = 0
        for i in range(len(predict)):
            # print(predict)
            if predict[i] == y_new[i]:
                count += 1

        acc = float(count/len(predict))
        # print('Accuracy: {}%'.format(acc*100))
        from sklearn.metrics import confusion_matrix

        confusion_matrix = confusion_matrix(
            y_true=y_new, y_pred=predict)

        # #Normalize CM
        confusion_matrix = cm = confusion_matrix.astype(
            'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(confusion_matrix)

        # plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(df_cm, cmap="coolwarm", annot=False)
        fig.set_size_inches(8, 6)
        ax.set_title("Confusion Matrix of RNN, Data: {}".format(filepath))
        ax.set_xlabel('Perdicted Label', fontsize=12)
        ax.set_ylabel('Actual Label', fontsize=12)

        plt.show()

        return predict, acc


# TENG().train()
# get_data_test()
score = {}
# filepath = ['test_0415.csv', 'test_0416.csv',
#             'test_0417.csv', 'test_0430.csv', 'test_1028.csv']
filepath = ['peak_20180115position_sum.csv', 'peak_20181118position_sum.csv']
for f in filepath:
    test, accuracy = TENG().forward_ni(f)
    score[f[5:9]] = accuracy
    print(f, accuracy)

###########################################################

# data = [x for x in range(1, 26)]
# for num in data:
#     print('Location: ', num)
#     filepath = './test/P{}.csv'.format(num)
#     test, accuracy = TENG().forward_EM(filepath, num)
#     score[num] = [accuracy*100]

df = pd.DataFrame.from_dict(data=score, orient='index')
print(df)
df.plot.bar(legend=False, rot=0, alpha=0.75)
plt.show()

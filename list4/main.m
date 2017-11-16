clear all;
close all;
clc;


# Loading data

load DadosLista4.mat;
X = Dados;
Y = y;

# Split data
[XTraining, YTraining, XTest, YTest] = split(X, Y, trainingRate=0.6);

naiveBayesModel = naiveBayes(XTraining, YTraining);


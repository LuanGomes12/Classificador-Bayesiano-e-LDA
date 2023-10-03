% ********** Luan Gomes Magalhães Lima - 473008                      ********** 
% ********** Tópicos Especiais em Telecomunicações 1 - Prática 4     **********

% Inicializações
clear all;
close all;
clc;

%% Ánalise exploratória e escolha dos atributos
% Carregar a base de dados
dados = readtable("BankNote_Authentication.csv");

% Propriedades da base
propriedades_base = dados.Properties;

% Entrada (features)
X = dados.Variables;
X = X(:, 1:end-1);

% Saída (targets)
y = dados.class;

% Considera-se que os atributos obedecem a uma distribuição gaussiana.
% 1º Atributo: Variância
y1 = dados.variance;
figure;
subplot(2, 2, 1);
histfit(y1);
title('ATB: Variance');
xlabel('Amostras');
ylabel('Frequência Relativa');

% 2º Atributo: Assimetria
y2 = dados.skewness;
subplot(2, 2, 2); 
histfit(y2);
title('ATB: Skewness');
xlabel('Amostras');
ylabel('Frequência Relativa');

% 3º Atributo: Curtose
y3 = dados.curtosis;
subplot(2, 2, 3); 
histfit(y3);
title('ATB: Curtosis');
xlabel('Amostras');
ylabel('Frequência Relativa');

% 4º Atributo: Entropia
y4 = dados.entropy;
subplot(2, 2, 4); 
histfit(y4);
title('ATB: Entropy');
xlabel('Amostras');
ylabel('Frequência Relativa');

% **** OBSERVAÇÕES ****
% O intuito é analisar visulamente se os atributos obedecem a uma
% distribuição gaussiana, uma possível análise visual é por meio dos
% histogramas mostrados.

%% Criação do Classificador Baysiano
% Foi utilizado o caso paramétrico para atributos contínuos

% Normalização dos dados
X = normalize(X);

% Garantir a aleatoriedade dos índices
ind_embaralhado = randperm(size(X,1));
X = X(ind_embaralhado, :);
y = y(ind_embaralhado);

% Divisão da base em conjunto de treinamento e teste - Relação 80/20
proprocao_treino = 0.8;
ind_treino = ind_embaralhado(1: round(proprocao_treino*size(X, 1)));
ind_teste = setdiff(ind_embaralhado, ind_treino);

% Conjunto de treinamento
X_train = X(ind_treino, :);
y_train = y(ind_treino);

% Conjunto de teste
X_test = X(ind_teste, :);
y_test = y(ind_teste);

% Matriz para armazenar as probabilidades de uma amostra pertencer a cada
% classe
matriz_prob = zeros(length(X_test), 2);

% Probabilidade a priori
P = zeros(1, 2);

% Probabilidade de cada classe
% Caso de uma base binária
for i = 1 : 2 
    % ETAPA DE TREINAMENTO
    % Probabilidade a priori da classe i
    elementos = find(y_train == i-1);
    qtd_elementos = length(elementos);
    P(1, i) = qtd_elementos/length(y_train);

    % Vetor de Médias
    M = zeros(1, size(X_train,2));
    for k = 1 : size(X_train,2)
        M(1, k) = mean(X_train(elementos, k));
    end

    % Matriz de Covariâncias
    X_atual = X_train(y_train == i-1, :);
    Cov = cov(X_atual);
    
    % ETAPA DE TESTE
    % Probabilidade de cada elemento do conjunto de teste pertencer a
    % classe i
    for j = 1 : length(X_test)
        amostra_atual = X_test(j, :);
        matriz_prob(j, i) =  calcular_discriminante(amostra_atual, M, Cov, P(1, i));
    end
end

% Acurácia
matriz_classes = zeros(size(X_test,1), 1);

for i = 1 : size(matriz_prob, 1)
    if min(matriz_prob(i, :)) == matriz_prob(i, 1)
        matriz_classes(i, 1) = 0;
    else
        matriz_classes(i, 1) = 1;
    end
end

acuracia = sum(matriz_classes == y_test)/length(X_test);
fprintf("Acurácia do Classificador Bayesiano: %.2f", acuracia*100);

%% Criação do Classificador LDA
% Divisão da base em K partes para implementação do K-Fold
K_fold = 10;
num_amostras_kfold = round(size(X, 1)/K_fold);

% Garantir a aleatoriedade dos índices
ind_embaralhado = randperm(size(X,1));
features = X(ind_embaralhado, :);
labels = y(ind_embaralhado);

% Matriz para armazenar as acurácias de cada K parte do K-Fold
acuracia = zeros(K_fold, 1);

% Loop do K-Fold
for j = 1 : K_fold
    % ETAPA DE TREINAMENTO
    % Divisão dos índices em teste e treino
    ind_teste = ind_embaralhado((j-1)*num_amostras_kfold+1 : j*num_amostras_kfold);
    ind_treino = setdiff(ind_embaralhado, ind_teste);

    % Conjunto de treinamento
    X_train = features(ind_treino, :);
    y_train = labels(ind_treino);

    % Conjunto de teste
    X_test = features(ind_teste, :);
    y_test = labels(ind_teste);

    % Implementação do LDA
    % Amostras da classe 1
    X1 = X_train(y_train == 0, :).';

    % Amostras da classe 2
    X2 = X_train(y_train == 1, :).';

    % Médias das classes
    u1 = mean(X1, 2); % Média da classe 1
    u2 = mean(X2, 2); % Média da classe 2

    % Matrizes de covariância
    % Necessário transpor antes de calcular a matriz de covariância
    S1 = cov(X1.');
    S2 = cov(X2.');

    % Matriz de dispersão dentro das classes
    Sw = S1 + S2;
    inv_Sw = inv(Sw);

    % Matriz de dispersão entre as classes
    SB = (u1 - u2) * (u1 - u2).';

    % Calcular os autovalores e autovetores
    [V, D] = eig(inv_Sw * SB);

    % Vetor de projeção ótimo
    w = V(:, 1);

    % Amostras projetadas
    y1 = w.' * X1;
    y2 = w.' * X2;

    % Médias projetadas
    u1p = w.' * u1;
    u2p = w.' * u2;
    
    % ETAPA DE TESTE
    % Limiar de separação
    lim = (u1p + u2p)/2;

    amostras_teste = X_test;
    proj_amostras_teste = w.' * amostras_teste';

    % Classificação com base no limiar encontrado
    predict = proj_amostras_teste < lim;
    predict = predict';

    % Quantidade de acertos
    acertos = 0;
    for i = 1 : length(predict)
        if predict(i) == y_test(i)
            acertos = acertos + 1;
        end
    end
    
    % Acurácia em cada K-Fold
    qtd_total_amostras = size(y_test);
    acuracia(j) = acertos/length(predict);
end

% Gráfico de dispersão
% Classe 1: referencia os elementos de valor 0
% Classe 2: referencia os elementos de valor 1
% Plotagem dos valores das amostras da classe 1, das amostras da classe 2,
% da média da classe 1 e da média da classe 2, respectivamente. Esses
% valores estão associados a última rodada do K-Fold. Serve apenas de
% parâmetro para analisar a dispersão dos dados, esses gráficos mudam a 
% cada iteração do K-Fold.
figure;
plot(X1(1,:), X1(2,:), 'r.', MarkerSize=10);
hold on, plot(X2(1,:), X2(2,:),'b.', MarkerSize=10);
hold on, plot(u1(1), u1(2),'g.', MarkerSize=40);
hold on, plot(u2(1), u2(2),'y.', MarkerSize=40);

% Reta passando pelo vetor de projeção para última rodada do K-Fold
x = -4:0.01:3;
g = w(2)/w(1)*x;
hold on, plot(x, g, 'k-')
grid on;
title('Banknote Authentication Dataset - LDA');
legend('Verdadeira', 'Falsa', 'Média da classe 1', 'Média da classe 2');

% Acurácia final do classificador LDA
acuracia_final = mean(acuracia);
fprintf("\nAcurácia do Classificador LDA: %.2f\n", acuracia_final*100);

%% Funções utilizadas
% Função discriminante utilizada no Classificador Bayesiano
function discriminante = calcular_discriminante(x, mu, mat_cov, p)
    discriminante = log(det(mat_cov)) + ((x-mu)*inv(mat_cov))*(x-mu)' - 2*log(p);
end
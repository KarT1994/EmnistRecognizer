import numpy as np
from collections import OrderedDict
from .cnnlib import Convolution,MaxPooling
from .common import ReLU, Affine, SoftmaxWithLoss, Dropout, BatchNormalization


class TwoConvNet7:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param2={'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 pool_param={'pool_size': 2, 'pad': 0, 'stride': 2},
                 pool_param2={'pool_size':2, 'pad':0, 'stride':2},
                 hidden_size=100,hidden_size2=1000, output_size=10, weight_init_std=0.01, weight_decay_lambda=0.01):
        """
        input_size :  入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param :  畳み込みの条件
        pool_param :  プーリングの条件
        hidden_size : 隠れ層のノード数
        output_size : 出力層のノード数
        weight_init_std ： 重みWを初期化する際に用いる標準偏差
        weight_decay_lambda：正則化の係数（L2ノルム）
        """
                
        # 一層目のフィルタ
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        input_size = input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size = (conv_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        pool_output_pixel = filter_num * pool_output_size * pool_output_size # プーリング後のピクセル総数

        # 二層目のフィルタ
        filter_num2 = conv_param2['filter_num']
        filter_size2 = conv_param2['filter_size']
        filter_pad2 = conv_param2['pad']
        filter_stride2 = conv_param2['stride']
        
        pool_size2 = pool_param2['pool_size']
        pool_pad2 = pool_param2['pad']
        pool_stride2 = pool_param2['stride']
        
        input_size2 = pool_output_size
        conv_output_size2 = (input_size2 + 2*filter_pad2 - filter_size2) // filter_stride2 + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size2 = (conv_output_size2 + 2*pool_pad2 - pool_size2) // pool_stride2 + 1 # プーリング後のサイズ(H,W共通)
        pool_output_pixel2 = filter_num2 * pool_output_size2 * pool_output_size2 # プーリング後のピクセル総数

        print(conv_output_size)
        print(pool_output_size)
        print(pool_output_pixel)
        print(conv_output_size2)
        print(pool_output_size2)
        print(pool_output_pixel2)


        # 重みの初期化
        self.params = {}
        std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda
        
        # W1は畳み込みフィルターの重み 
        # 配列形状=(フィルター枚数, チャンネル数, フィルター高さ, フィルター幅)
        self.params['W1'] = std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['W2'] =  std * np.random.randn( filter_num, filter_num, filter_size,filter_size)

        # b1は畳み込みフィルターのバイアス
        # 配列形状=(フィルター枚数)
        self.params['b1'] = np.zeros(filter_num)
        self.params['b2'] = np.zeros(filter_num)

        # 全結合層の重みW
        # 配列形状=(前の層のノード数, 次の層のノード数)         
        self.params['W3'] = std * np.random.randn(filter_num2, filter_num, filter_size2, filter_size2)
        self.params['W4'] = std * np.random.randn( filter_num2, filter_num2, filter_size2,filter_size2)
        
        # 全結合層のバイアスb
        # 配列形状=(次の層のノード数)        
        self.params['b3'] = np.zeros(filter_num2)
        self.params['b4'] = np.zeros(filter_num2)

        # 全結合層の重みW
        # 配列形状=(前の層のノード数, 次の層のノード数)        
        self.params['W5'] = std * np.random.randn(pool_output_pixel2, hidden_size )
        # 全結合層のバイアスb
        # 配列形状=(次の層のノード数)        
        self.params['b5'] = np.zeros(hidden_size)
        # 全結合層の重みW
        # 配列形状=(前の層のノード数, 次の層のノード数)        
        self.params['W6'] = std * np.random.randn(hidden_size, hidden_size2 )
        # 全結合層のバイアスb
        # 配列形状=(次の層のノード数)        
        self.params['b6'] = np.zeros(hidden_size2)
        # 全結合層の重みW
        # 配列形状=(前の層のノード数, 次の層のノード数)        
        self.params['W7'] = std * np.random.randn(hidden_size2, output_size )
        # 全結合層のバイアスb
        # 配列形状=(次の層のノード数)        
        self.params['b7'] = np.zeros(output_size)

        self.params['gamma0'] = np.ones(1)
        self.params['beta0'] = np.zeros(1)
        self.params['gamma1'] = np.ones(filter_num)
        self.params['beta1'] = np.zeros(filter_num)
        self.params['gamma2'] = np.ones(filter_num2)
        self.params['beta2'] = np.zeros(filter_num2)
        self.params['gamma3'] = np.ones(hidden_size)
        self.params['beta3'] = np.zeros(hidden_size) 
        self.params['gamma4'] = np.ones(hidden_size2)
        self.params['beta4'] = np.zeros(hidden_size2) 

        # レイヤの生成
        # VGG風の構成（二重の畳み込み層＋プーリング層を２ブロック）
        # 畳み込み層のドロップアウトはバッチ正規化と相性が悪いため、不採用
        self.layers = OrderedDict()
        self.layers["Batch0"] = BatchNormalization(gamma=self.params['gamma0'], beta=self.params['beta0'])
        self.layers['Conv11'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])  
        self.layers['Relu1'] = ReLU()
        #self.layers['Dropout0'] = Dropout(0.25)
        self.layers['Conv12'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride'], conv_param['pad'])          
        self.layers["Batch1"] = BatchNormalization(gamma=self.params['gamma1'], beta=self.params['beta1'])
        self.layers['Relu2'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        #self.layers['Dropout1'] = Dropout(0.25)
        self.layers['Conv21'] = Convolution(self.params['W3'], self.params['b3'],
                                           conv_param2['stride'], conv_param2['pad'])  
        self.layers['Relu3'] = ReLU()
        #self.layers['Dropout9'] = Dropout(0.25)
        self.layers['Conv22'] = Convolution(self.params['W4'], self.params['b4'],
                                           conv_param2['stride'], conv_param2['pad']) 
                
        self.layers["Batch2"] = BatchNormalization(gamma=self.params['gamma2'], beta=self.params['beta2'])
        self.layers['Relu4'] = ReLU()
        self.layers['Pool2'] = MaxPooling(pool_h=pool_size2, pool_w=pool_size2, stride=pool_stride2, pad=pool_pad2)
        #self.layers['Dropout2'] = Dropout(0.25)

        self.layers['Affine1'] = Affine(self.params['W5'], self.params['b5'])
        self.layers["Batch3"] = BatchNormalization(gamma=self.params['gamma3'], beta=self.params['beta3'])
        self.layers['ReLU5'] = ReLU()
        self.layers['Dropout3'] = Dropout(0.5)
        self.layers['Affine2'] = Affine(self.params['W6'], self.params['b6'])
        self.layers["Batch4"] = BatchNormalization(gamma=self.params['gamma4'], beta=self.params['beta4'])
        self.layers['ReLU6'] = ReLU()
        self.layers['Dropout3'] = Dropout(0.5)
        self.layers['Affine3'] = Affine(self.params['W7'], self.params['b7'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        """
        予測（損失関数は除く）
        入力：予測元データ（画像）
        戻値：予測結果
        """
        for layer in self.layers.values():
            #print(layer)
            x = layer.forward(x, train_flg)

        return x

    def loss(self, x, t, train_flg=False):
        """
        損失算出
        入力：予測結果、正解値
        戻値：損失
        """
        y = self.predict(x, train_flg)
        lmd = self.weight_decay_lambda        
        weight_decay = 0
        for idx in range(1, 3 + 2):
            W = self.params['W' + str(idx)]
            
            # 全ての行列Wについて、1/2* lambda * Σwij^2を求め、積算していく
            weight_decay += 0.5 * lmd * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t, train_flg=False, batch_size=100):
        """
        精度算出
        入力：予測結果、正解値、train_flgは訓練時のみTrue、batchサイズ）
        出力：精度
        """
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t, train_flg=False):
        """
        勾配
        入力：予測元データ（画像）
        出力：勾配
        """
        # forward
        self.loss(x, t, train_flg)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        lmd = self.weight_decay_lambda
        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv11'].dW + lmd * self.layers['Conv11'].W, self.layers['Conv11'].db
        grads['W2'], grads['b2'] = self.layers['Conv12'].dW + lmd * self.layers['Conv12'].W, self.layers['Conv12'].db
        grads['W3'], grads['b3'] = self.layers['Conv21'].dW + lmd * self.layers['Conv21'].W, self.layers['Conv21'].db
        grads['W4'], grads['b4'] = self.layers['Conv22'].dW + lmd * self.layers['Conv22'].W, self.layers['Conv22'].db
        grads['W5'], grads['b5'] = self.layers['Affine1'].dW + lmd * self.layers['Affine1'].W, self.layers['Affine1'].db
        grads['W6'], grads['b6'] = self.layers['Affine2'].dW + lmd * self.layers['Affine2'].W, self.layers['Affine2'].db
        grads['W7'], grads['b7'] = self.layers['Affine3'].dW + lmd * self.layers['Affine3'].W, self.layers['Affine3'].db
        grads['gamma0'], grads['beta0'] = self.layers['Batch0'].dgamma, self.layers['Batch0'].dbeta
        grads['gamma1'], grads['beta1'] = self.layers['Batch1'].dgamma, self.layers['Batch1'].dbeta
        grads['gamma2'], grads['beta2'] = self.layers['Batch2'].dgamma, self.layers['Batch2'].dbeta
        grads['gamma3'], grads['beta3'] = self.layers['Batch3'].dgamma, self.layers['Batch3'].dbeta
        grads['gamma4'], grads['beta4'] = self.layers['Batch4'].dgamma, self.layers['Batch4'].dbeta


        return grads


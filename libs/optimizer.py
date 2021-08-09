# coding: utf-8
import numpy as np


class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr=0.01):
        """
        lr : 学習係数 learning rate
        """
        self.lr = lr
        
    def update(self, params, grads):
        """
        重みの更新
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
class NesterovAG:
    """
    Nesterov Accelerated Gradient
    """
    def __init__(self, lr=0.01, momentum=0.9):
        """
        lr : 学習係数 learning rate
        momentm : モーメンタム係数
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.v is None:
            """
            初回のみ
            """
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        # 重みを更新
        for key in params.keys():
            v_pre = self.v[key]
            self.v[key] = v_pre * self.momentum - self.lr * grads[key]
            params[key] += -self.momentum* v_pre + (self.momentum+1) * self.v[key]


class RMSProp:
    """
    RMSProp
    """
    def __init__(self, lr=0.01, rho=0.9):
        """
        lr : 学習係数 learning rate
        rho : 減衰率
        """
        self.lr = lr
        self.h = None
        self.rho = rho
        self.epsilon = 1e-6
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.h is None:
            """
            初回のみ
            """
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key] * grads[key]          
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + self.epsilon) ) # 原論文に合わせてepsilonをルートの中に入れる



class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            grads[key] += self.v[key]

class Adam:
    """
    Adam
    """
    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.iter = 0
        self.m = None # 1次モーメント. 勾配の平均に相当する
        self.v = None  # 2次モーメント. 勾配の分散に相当する(中心化されていない)
        self.epsilon = 1e-8
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        
        for key in params.keys():
            self.m[key] = self.rho1*self.m[key] + (1-self.rho1)*grads[key] # 1次モーメント
            self.v[key] = self.rho2*self.v[key] + (1-self.rho2)*(grads[key]**2) #2次モーメント
            
            # モーメントのバイアス補正
            # 計算初期の頃のモーメントを補正することが目的
            # 計算が進行する(self.iterが大きくなる)と、分母は1に近づく
            m = self.m[key] / (1 - self.rho1**self.iter) # 1次モーメント
            v = self.v[key] / (1 - self.rho2**self.iter) # 2次モーメント
            
            # 重みの更新
            params[key] -= self.lr * m / (np.sqrt(v) + self.epsilon)
            


class Adadelta:
    """
    Adadelta
    """
    def __init__(self, rho=0.95):
        """
        rho : 減衰率
        """
        self.h = None
        self.r = None        
        self.rho = rho
        self.epsilon = 1e-6
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.h is None:
            """
            初回のみ
            """
            self.h = {}
            self.r = {}            
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.r[key] = np.zeros_like(val)
                
        for key in params.keys():
            
            # 1ステップ前における更新量の2乗の移動平均のルートを求める
            rms_param = np.sqrt(self.r[key] + self.epsilon)
                        
            # 勾配の2乗の移動平均を求める
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key] * grads[key]  
            
            # 勾配の2乗の移動平均のルートを求める
            rms_grad =  np.sqrt(self.h[key] + self.epsilon)
            
            # 更新量の算出
            dp = - (rms_param / rms_grad) * grads[key]             
            
            # 重みの更新
            params[key] += dp
            
            # 次ステップのために、更新量の2乗の移動平均を求める 
            self.r[key]  = self.rho * self.r[key] + (1 - self.rho) * dp * dp
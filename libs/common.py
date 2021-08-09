import numpy as np

def cross_entropy_error(y, t):
    """
    引数：y 予測値, t 正解値 
    戻値：クロスエントロピー誤差  
    """
    # 予測値が１つだけのとき行列を反転させる
    if y.ndim==1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
        
    batch_size = y.shape[0]
    delta = 1e-7    # 0割り防止
    return -np.sum( t * np.log(y + delta)) / batch_size # batchサイズで損失は平均化する

def softmax(x):
    """
    引数：前層の出力
    戻値：０～１の確率（合計１になる）
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


class ReLU:
    def __init__(self):
        self.mask = None # 0以上をTrueにするフィルタ 
        
    def forward(self, x, train_flg=False):
        """
        引数：前層の出力
        戻値：1 (x>0), 0(x<=0)
        """
        self.mask = (x <= 0)
        out = x.copy() # xを書き換えないようにコピーで渡す（参照渡しはNG）
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        """
        入力：後層の勾配
        戻値：勾配
        """        
        dout[self.mask] = 0
        dLdx = dout
        return dLdx
    
class Affine:
    def __init__(self, W, b):
        self.W =W   # 重み
        self.b = b  # バイアス
        
        self.x = None   # 前層の出力
        self.original_x_shape = None    # 入力データの形状
        self.dW = None  # 重みの勾配
        self.db = None  # バイアスの勾配

    def forward(self, x, train_flg=False):
        """
        引数：前層の出力
        戻値：重み*入力+バイアス
        """
        # 画像形式にreshape
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """
        引数：後層の勾配
        戻値：勾配
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape) 
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 正解値

    def forward(self, x, t):
        """
        引数：前層の出力
        戻値：損失
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        """
        引数：後層の勾配（通常は最終層なので１）
        戻値：勾配
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size # 平均化されている損失をデータ毎の損失にするためbatchサイズで割る

        return dx


class Dropout:
    def __init__(self, ratio):
        self.ratio = ratio  # 無効化するノードの比率（0 ~ 1)
        self.mask = None    # フィルタ
    
    def forward(self, x, train_flg=False):
        if train_flg == True:
            self.mask = (np.random.rand(*x.shape) > self.ratio) # 一律分布から０～１をサンプリング
            return x * self.mask
        else:
            return (1 - self.ratio) * x

    def backward(self, dout):
        return dout * self.mask




class BatchNormalization:
    def __init__(self, gamma, beta, rho=0.9, moving_mean=None, moving_var=None):
        self.gamma = gamma # スケールさせるためのパラメータ, 学習によって更新させる.
        self.beta = beta # シフトさせるためのパラメータ, 学習によって更新させる
        self.rho = rho # 移動平均を算出する際に使用する係数

        # 予測時に使用する平均と分散
        self.moving_mean = moving_mean   # muの移動平均
        self.moving_var = moving_var     # varの移動平均
        
        # 計算中に算出される値を保持しておく変数群
        self.batch_size = None
        self.x_mu = None
        self.x_std = None        
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        """
        順伝播計算
        x :  CNNの場合は4次元、全結合層の場合は2次元  
        """
        if x.ndim == 4:
            """
            画像形式の場合
            """
            N, C, H, W = x.shape
            #print("x.shape{}" .format(x.shape))
            x = x.transpose(0, 2, 3, 1) # NHWCに入れ替え
            x = x.reshape(N * H * W, C)  # (N*H*W,C)の2次元配列に変換
            out = self.__forward(x, train_flg)
            out = out.reshape(N, H, W, C)# 4次元配列に変換
            out = out.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif x.ndim == 2:
            """
            画像形式以外の場合
            """
            out = self.__forward(x, train_flg)           
            
        return out
            
    def __forward(self, x, train_flg, epsilon=1e-8):
        """
        x : 入力. N×Dの行列. Nはバッチサイズ. Dは手前の層のノード数
        """
        N, D = x.shape
        if (self.moving_mean is None) or (self.moving_var is None):
            self.moving_mean = np.ones(D)
            self.moving_var = np.zeros(D)
                        
        if train_flg:
            """
            学習時
            """
            # 入力xについて、Nの方向に平均値を算出. 
            mu_0 = np.mean(x, axis=0) # 要素数D個のベクトル
            mu = np.broadcast_to(mu_0, (N, D)) # Nの方向にブロードキャスト
            
            # 入力xから平均値を引く
            x_mu = x - mu   # N×D行列
            
            # 入力xの分散を求める
            var = np.mean(x_mu**2, axis=0)  # 要素数D個のベクトル

            # 入力xの標準偏差を求める(epsilonを足してから標準偏差を求める)
            std = np.sqrt(var + epsilon)  # 要素数D個のベクトル
            
            # 標準偏差の逆数を求める
            std_inv = 1 / std
            std_inv = np.broadcast_to(std_inv, (N, D)) # Nの方向にブロードキャスト

            # 標準化
            x_std = x_mu * std_inv  #N*D行列
                  
            # 値を保持しておく
            self.batch_size = x.shape[0]
            self.x_mu = x_mu
            self.x_std = x_std
            self.std = std
            self.moving_mean = self.rho * self.moving_mean + (1-self.rho) * mu_0
            self.moving_var = self.rho * self.moving_var + (1-self.rho) * var            
        else:
            """
            予測時
            """
            x_mu = x - np.broadcast_to(self.moving_mean, (N, D)) # Nの方向にブロードキャスト
            x_std = x_mu / np.sqrt(np.broadcast_to(self.moving_var + epsilon, (N, D))) # N*D行列
            
        # gammaでスケールし、betaでシフトさせる
        #print(self.gamma.shape)
        #print(x_std.shape)
        #print(self.beta.shape)
        out = self.gamma * x_std + self.beta # N*D行列
        return out

    def backward(self, dout):
        """
        逆伝播計算
        dout : CNNの場合は4次元、全結合層の場合は2次元  
        """
        if dout.ndim == 4:
            """
            画像形式の場合
            """            
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1) # NHWCに入れ替え
            dout = dout.reshape(N*H*W, C) # (N*H*W,C)の2次元配列に変換
            dx = self.__backward(dout)
            dx = dx.reshape(N, H, W, C)# 4次元配列に変換
            dx = dx.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif dout.ndim == 2:
            """
            画像形式以外の場合
            """
            dx = self.__backward(dout)

        return dx

    def __backward(self, dout):
        """
        ここを完成させるには、計算グラフを理解する必要があり、実装にかなり時間がかかる.
        """
        N, D = self.x_mu.shape
        
        # betaの勾配
        dbeta = np.sum(dout, axis=0)
        
        # gammaの勾配(Nの方向に合計)
        dgamma = np.sum(self.x_std * dout, axis=0)
        
        # Xstdの勾配
        a1 = self.gamma * dout
        
        # Xmuの勾配(1つ目)
        a2 = a1 / self.std
        
        # 標準偏差の逆数の勾配
        a3 = a1 * self.x_mu
        a3 = np.sum(a3, axis=0) # Nの方向に合計
        
        # 標準偏差の勾配
        a4 = -(a3) / (self.std * self.std)
        
        # 分散の勾配
        a5 = 0.5 * a4 / self.std
        
        # Xmuの2乗の勾配
        a6 = a5 / self.batch_size
        a6 = np.broadcast_to(a6, (N, D)) # Nの方向にブロードキャスト
        
        # Xmuの勾配(2つ目)
        a7 = 2.0  * self.x_mu * a6
        
        # muの勾配
        a8 = -(a2+a7)
        a8 = np.sum(a8, axis=0) # Nの方向に合計

        # Xの勾配
        a9 = a8 / self.batch_size
        a9 = np.broadcast_to(a9, (N, D)) # Nの方向にブロードキャスト
        dx = a2 + a7 + a9
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
                
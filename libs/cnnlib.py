import numpy as np

def im2col(input_data, fh, fw, stride=1, pad=1, pad_val=0):
    '''
    input_data = 四次元の画像（画像数、チャネル、高さ、幅）
    fh = フィルターの高さ
    fw =　フィルターの幅
    pad = パッドの埋める範囲(0はパッドしない)
    pad_val = padで埋める値
    '''

    #画像の情報を取得
    N, C, H, W = input_data.shape
    
    #出力ブロックの高さoh,と幅ow
    oh =  (H + 2*pad - fh) // stride + 1
    ow = (W + 2 * pad - fw) // stride + 1
    
    # 入力画像のパディング処理
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=pad_val)
    col = np.zeros((N, C, fh, fw, oh, ow))
    
    for y in range(fh):
        y_max = y + stride * oh # y_max - 1が配列の最後の要素になるので、一回分多めにとる oh - 1 + 1 = oh
        for x in range(fw):
            x_max = x + stride * ow
            col[:,:, y, x,:,:] = img[:,:, y:y_max:stride, x:x_max:stride]
            
    col = col.transpose(0, 4, 5, 1, 2, 3)
    col = col.reshape(N * oh * ow, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0, is_backward=False):
    """
    Parameters
    ----------
    col : 2次元配列
    input_shape : 入力データの形状,  (データ数, チャンネル数, 高さ, 幅)の4次元配列
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    return : (データ数, チャンネル数, 高さ, 幅)の4次元配列. 画像データの形式を想定している
    -------
    """
    
    # 入力画像(元画像)のデータ数, チャンネル数, 高さ, 幅を取得する
    N, C, H, W = input_shape
    
    # 出力(畳み込みまたはプーリングの演算後)の形状を計算する
    out_h = (H + 2*pad - filter_h)//stride + 1 # 出力画像の高さ(端数は切り捨てる)
    out_w = (W + 2*pad - filter_w)//stride + 1 # 出力画像の幅(端数は切り捨てる)
    
    # 配列の形を変えて、軸を入れ替える
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 配列の初期化
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    # 2*padは、pad分を大きくとっている。
    # (stride - 1)は、im2colで画像が切り捨てられる場合のサイズ調整分。
    # im2colで、strideを2以上に設定した場合、あるひとつの方向に最大で(stride - 1)個の画素が切り捨てられる。
    # このような切り捨てが発生する場合、col2imのimg[:, :, y:y_max:stride, x:x_max:stride] でスライスを使って
    # stride刻みで値を代入する際にエラーが出て止まってしまう。そのため、縦横ともに(stride - 1)個分だけ余分に配列を確保しておき、
    # 最後に余分なところを切り捨てることでサイズを調整している。
    
    # 配列を並び替える
    for y in range(filter_h):
        """
        フィルターの高さ方向のループ
        """        
        y_max = y + stride*out_h
        for x in range(filter_w):
            """
            フィルターの幅方向のループ
            """            
            x_max = x + stride*out_w
            
            # colから値を取り出し、imgに入れる
            if is_backward:
                """
                逆伝播計算の場合
                """
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :] # 伝わってきた勾配を足していく
            else:
                """
                元のimに戻ることを確認したい場合
                """
                img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]
                
    return img[:,:, pad:H + pad, pad:W + pad]  # pad分は除いておく(pad分と(stride - 1)分を除いて真ん中だけを取り出す)
    


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        # 学習パラメータ
        self.W = W
        self.b = b
        # 畳み込みパラメータ
        self.stride = stride
        self.pad = pad
        # 逆伝播（勾配）
        self.dW = None
        self.db = None
        #逆伝播に必要な計算
        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x, train_flg=False):
        """
        畳み込み順伝播
        引数 x (N,C,H,Wの四次元画像)
        """
        FN,C,FH, FW = self.W.shape
        N, C, H, W = x.shape
        oh = (H + 2*self.pad -FH)//self.stride + 1
        ow = (W + 2*self.pad -FW)//self.stride + 1        

        col = im2col(x, FH, FW, stride=self.stride, pad=self.pad)
        col_W = self.W.transpose(1, 0, 2, 3).reshape(C * FH * FW, -1)
        
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N,oh, ow, -1).transpose(0,3,1,2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        """
        畳み込みの逆伝播
        引数 dout （本クラスに入力される微分）
        """
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1,FN)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad, is_backward=True)

        self.dcol = dcol # 結果を確認するために保持しておく
            
        return dx


class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        # プーリングのパラメータ
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # 逆伝播で使用するパラメータ
        self.dcol = None
        self.col = None
        self.arg_max = None
        self.x = None

    def forward(self, x, train_flg=False):
        """
        マックスプーリングの順伝播
        引数　四次元画像（N,C,H,W）
        """
        N, C, H, W = x.shape
        oh = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        ow = (W + 2 * self.pad - self.pool_w) // self.stride + 1
        col = im2col(x, self.pool_h, self.pool_w, stride=self.stride, pad=self.pad, pad_val= -np.inf)
        col = col.reshape(N * oh * ow * C, self.pool_h * self.pool_w)
        
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, oh, ow, C).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.arg_max = arg_max

        return out
    
    def backward(self, dout):
        """
        逆伝播計算
        マックスプーリングでは、順伝播計算時に最大値となった場所だけに勾配を伝える
        順伝播計算時に最大値となった場所は、self.arg_maxに保持されている        
        dout : 出力層側から伝わってきた勾配
        return : 入力層側へ伝える勾配
        """        
        
        # doutのチャンネル数軸を4番目に移動させる
        dout = dout.transpose(0, 2, 3, 1)
        
        # プーリング適応領域の要素数(プーリング適応領域の高さ × プーリング適応領域の幅)
        pool_size = self.pool_h * self.pool_w
        
        # 勾配を入れる配列を初期化する
        # dcolの配列形状 : (doutの全要素数, プーリング適応領域の要素数) 
        # doutの全要素数は、dout.size で取得できる
        dcol = np.zeros((dout.size, pool_size))
        
        # 順伝播計算時に最大値となった場所に、doutを配置する
        # dout.flatten()でdoutを1次元配列に変形できる
        dcol[np.arange(dcol.shape[0]), self.arg_max] = dout.flatten()
        
        # 勾配を4次元配列(データ数, チャンネル数, 高さ, 幅)に変形する
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad, is_backward=True)
        
        self.dcol = dcol # 結果を確認するために保持しておく
        
        return dx
        

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class encoderCNN(nn.Module):
    
    def __init__(self, params):
        super(encoderCNN, self).__init__()

        # the pretrained RenNet-152
        resnet = models.resnet152( pretrained=True )

        self.conv_layers = nn.Sequential( *list(resnet.children() )[:-2] ) # returns A = [a_1, ..., a_k]
        self.pool = nn.AvgPool2d(7) # returns a_g = sum(a)/k
        self.feature_i = nn.Linear( 2048, params.hidden_dim ) # transform to local image feature vectors v_1 .. v_k
        self.feature_g = nn.Linear( 2048, params.embed_dim ) # transform to global image feature vector v_g

        self.relu = nn.ReLU()
    
    def forward(self, img):
        
        A = self.conv_layers(img)
        
        a_g = self.pool(A)
        a_g = a_g.view(a_g.size(0), -1)
        v_g = self.relu( self.feature_g(a_g) )

        A = A.view(A.size(0), A.size(1), -1).transpose(1,2)
        V = self.relu( self.feature_i(A) )

        return V, v_g



class LSTM_cell(nn.Module):

    def __init__(self, params):
        super(LSTM_cell, self).__init__()

        # LSTM cell
        self.LSTM_cell = nn.LSTM( params.embed_dim * 2, params.hidden_dim, 1, batch_first=True)

        # Visual sentinel
        self.W_x = nn.Linear( params.embed_dim * 2, params.hidden_dim , bias=False)
        self.W_h = nn.Linear( params.hidden_dim, params.hidden_dim , bias = False)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    
    def forward(self, x_t, states, h=None):

        h_t, states = self.LSTM_cell( x_t, states )

        if h == None:
            if torch.cuda.is_available():
                h = torch.zeros( h_t.size() ).cuda()
            else:
                h = torch.zeros( h_t.size() )
        g_t = self.sigmoid( self.W_x(x_t) + self.W_h(h) )
        mem = states[1].permute(1, 0, 2)
        s_t = g_t * self.tanh( mem )
        
        return s_t, h_t, states


class Atten(nn.Module):

    def __init__(self, params):
        super(Atten, self).__init__()

        # z_t
        self.W_v = nn.Linear( params.hidden_dim, 49, bias=False )
        self.W_g = nn.Linear( params.hidden_dim, 49, bias=False )
        self.W_h = nn.Linear( 49, 1, bias=False )

        # hat(z_t) : hat_z
        self.W_s = nn.Linear( params.hidden_dim, 49, bias=False )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
    
    def forward(self, V, s_t, h_t):
        
        z_t = self.W_v( V ).unsqueeze(1) + self.W_g( h_t ).unsqueeze(2)
        z_t = self.tanh( z_t )
        z_t = self.W_h( z_t ).squeeze(3)

        alpha = self.softmax( z_t.view( -1, z_t.size(2) ) )
        alpha = alpha.view( z_t.size(0), z_t.size(1), -1 )

        c_t = torch.bmm( alpha, V ).squeeze(2)

        hat_z = self.W_s( s_t ) + self.W_g( h_t )
        hat_z = self.W_h( self.tanh( hat_z ) )

        # z_t = [ batch_size, 1, 49 ]
        # hat_z = [ batch_size, batch_size, 1 ]
        cat = torch.cat( (z_t, hat_z), dim=2 )
        cat = cat.view( -1, cat.size(2) )
        hat_alpha = self.softmax( cat ).view(cat.size(0), cat.size(1), -1)
        beta = alpha[:, :, -1].unsqueeze(2)

        hat_c = beta * s_t + (1-beta) * c_t

        return hat_c



class decoderLSTM(nn.Module):

    def __init__(self, params):
        super(decoderLSTM, self).__init__()

        self.embedding = nn.Embedding( params.vocab_sz, params.embed_dim )

        self.LSTM_cell = LSTM_cell(params)
        self.Atten = Atten(params)
        self.MLP = nn.Linear( params.hidden_dim, params.vocab_sz )

        self.vocab_sz  = params.vocab_sz

    def forward(self, V, v_g, caption):
        
        w = self.embedding( caption )
        x = torch.cat( (w, v_g.unsqueeze(1).expand_as(w)), dim=2 )

        if torch.cuda.is_available():
            preds = Variable( torch.zeros( x.size(0), x.size(1), self.vocab_sz ).cuda() )
        else:
            preds = Variable( torch.zeros( x.size(0), x.size(1), self.vocab_sz ) )

        h_t = None
        states = None
        for t in range(x.size(1)):
            x_t = x[:, t, :].unsqueeze(1)
            
            s_t, h, states = self.LSTM_cell(x_t, states, h_t)
            h_t = h
            c_t = self.Atten(V, s_t, h_t)

            pred = self.MLP(c_t + h_t)
            preds[:, t, :] = pred[:, 0, :]
        
        return preds


class AdaptiveAttention(nn.Module):
    
    def __init__(self, params):
        super(AdaptiveAttention, self).__init__()

        self.encoder = encoderCNN(params)
        self.decoder = decoderLSTM(params)

    def forward(self, imgs, captions, l):
        V, v_g = self.encoder(imgs)
        preds = self.decoder(V, v_g, captions)

        preds = pack_padded_sequence(preds, l, batch_first=True)

        return preds
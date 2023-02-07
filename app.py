import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tkinter import Tk, Frame, Label, Button, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# defining global variables
TITLE = 'cDNA Oral Cancer Classifier'
HEIGHT=650
WIDTH=580
bg = 'white'

# utility functions
def hght_percent(perc):
    return (HEIGHT / 100) * perc

def width_percent(perc):
    return (WIDTH / 100) * perc


# wrapper for sci-kit learn model object
#   self.model: Sci-kit learn model object
#   self.data: pd.Dataframe object
class Model:
    
    def __init__(self, model, data):
        self._model_obj = model
        self._data = data
        self._genes = list(data.columns)
        self._labels = data['label']
        self._features = data.drop(['label'], axis=1)
        self._sample = None

# tKinter application
#   self.root: Tk()

class App:
    def __init__(self, root):
        self._root = root
        self._root.title(TITLE)
        self._root.geometry(f'{WIDTH}x{HEIGHT}')
        self._root.configure(bg=bg)
        self._root.resizable(False, False)
        self._model = self.load_model_data('svm_model.pkl', 'data_obj.pkl')
        self._plot = self.plot_model(self._model._data)
        self.curr_file_ = None
        
        self._frame = Frame(self._root, bg=bg, height=hght_percent(20), width=width_percent(80), highlightbackground="black", highlightthickness=4)
        self._frame.place(x= width_percent(10), y=hght_percent(70))
        self.pred_label = Label(self._frame, bg=bg, font=("Arial",32))
        self.pred_label.place(x=width_percent(22), y=20)
        self.file_label = Label(self._frame, bg=bg, font=("Arial",12))
        self.file_label.place(x=width_percent(18), y=90)
        Label(self._root, bg=bg, text='TP: 31',font=("Arial",16)).place(x=width_percent(10), y=hght_percent(52))
 
        Label(self._root, bg=bg, text='TN: 34',font=("Arial",16)).place(x=width_percent(30), y=hght_percent(52))

        Label(self._root, bg=bg, text='FN: 4',font=("Arial",16)).place(x=width_percent(55), y=hght_percent(52))
        Label(self._root, bg=bg, text='FP: 7',font=("Arial",16)).place(x=width_percent(78), y=hght_percent(52))
        
        Button(root, width = 9, height = 1, text = "Import CSV", command = self.import_sample).place(x = width_percent(30)-3, y = hght_percent(60))
        Button(root, width = 5, height = 1, text = "Predict", command= self.prediction).place(x = width_percent(60)-3, y = hght_percent(60)) # Predict Button

    # imports and cleans cDNA data of biopsy sample
    def import_sample(self, event=None):
        if self._model._sample is not None:
            self._model._data.drop(self._model._data.tail(1).index, inplace=True)
        self.curr_file_ = filedialog.askopenfilename()
        load_data = pd.read_csv(self.curr_file_, sep='\t')
        columns = [str(ref) for ref in load_data.ID_REF]
        sample_df = pd.DataFrame([list(load_data.VALUE)], columns=columns)
        sample_df = sample_df[sample_df.columns.intersection(self._model._genes)]
        self._model._sample = sample_df
        print('Sample Gene Expressions: \n', self._model._sample)
        sample_df['label'] = 0.5
        self._model._data = pd.concat((self._model._data, sample_df), axis=0)


    # loads custom Model obj
    #   model_file: Pickle object
    #   data_file: Pickle object
    #   returns: custom Model object
    def load_model_data(self, model_file, data_file):
        df = pd.read_pickle(data_file)
        with open(model_file, 'rb') as f:
            skmodel_obj = pickle.load(f)
        model = Model(skmodel_obj, df)
        return model
    
    # applies loaded sample to model for prediction and calls self.plot_model
    def prediction(self):
        sample = self._model._sample.drop(['label'], axis=1)
        data = self._model._data
        features = np.array(sample)
        self.file_label['text'] = self.curr_file_[-26:]
        prediction = self._model._model_obj.predict(features)
        
        if prediction == 0:
            self._plot=self.plot_model(data)
            self.pred_label['text'] = 'Negative'
            return
        else:
            self._plot=self.plot_model(data)
            self.pred_label['text'] = 'Positive'
            return
    
    # performs tSNE upon prediction training data and plots results
    #   data: pd.Dataframe object
    def plot_model(self, data):
        X = np.array(data.drop(['label'], axis=1))
        y = np.array(data['label'])
        
        tSNE_data = TSNE(n_components=3, random_state=99).fit_transform(X)
        tSNE_x, tSNE_y, tSNE_z = list(zip(*tSNE_data))

        tSNE_df = pd.DataFrame(tSNE_data, columns=['tSNE_X', 'tSNE_Y', 'tSNE_Z'])
        tSNE_df['label'] = y

        tSNE_fig = plt.figure(figsize=(4,3))
        ax1 = tSNE_fig.add_axes([0,0,1,1],projection='3d')
        ax1.set_xlabel('tSNE-1')
        ax1.set_ylabel('tSNE-2')
        ax1.set_zlabel('tSNE-3')
        ax1.scatter3D(tSNE_x, tSNE_y, tSNE_z, s=30, c=y, cmap='jet')
        tSNE_plot = FigureCanvasTkAgg(tSNE_fig, self._root)
        tSNE_plot.get_tk_widget().place(x=70,y=0)

# initialze app and run main loop
def main():
    root = Tk()
    gui = App(root)
    gui._root.mainloop()
    return
        
if '__main__' == __name__:
    main()
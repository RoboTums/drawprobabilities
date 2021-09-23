import flask
from flask import Flask, render_template, url_for, request
import base64
import numpy as np
import cv2
import pandas as pd
from sympy import divisors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats
import json
import scipy.stats as sp
import fitter
import plotly
import matplotlib.pyplot as plt
# Initialize the useless part of the base64 encoded image.
init_Base64 = 21

# Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')


def process_image(im):
    X = []
    Y = []
    pd.DataFrame(im).to_csv('debug_raw.csv')
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] != 0:
                X.append(i)
                Y.append(j)
    point_df = pd.DataFrame([X, Y]).T
    point_df.columns = ['Y', "X"]
    point_df.X = (point_df.X - 2*point_df.X) + im.shape[1]
    point_df.Y =  -point_df.Y +400
    point_df.to_csv('debug.csv')
    return point_df


def point_table_to_hist(point_df, num_bins):
    point_df.sort_values(by='X',inplace=True)
    heights = []
    x_range = [0]
    points_per_bin = point_df.shape[0]//(num_bins)
    for bin_i in range(points_per_bin, point_df.shape[0], points_per_bin):
        last_i = max(0, bin_i - points_per_bin)
        print(point_df.iloc[last_i:bin_i, :].X.median())
        x_range.append(point_df.iloc[last_i:bin_i, :].X.median())
        height = point_df.iloc[last_i:bin_i, :].Y.mean()
        heights.append(height)
    return np.array(heights), np.array(x_range) #np.linspace(point_df.X.min(), point_df.X.max(), num_bins)


def predict_image(point_df):
    # you have to renomalize this or you wont get anywhere near a PDF
    point_df['Y'] = point_df['Y']/point_df['Y'].sum()
    # best bin divisor is the largest divisor. if prime then we drop the last row and do the same.
    if point_df.shape[0] % 2 == 0 and point_df.shape[0] >= 4:
        num_bins = divisors(point_df.shape[0])[-2]
    else:
        point_df.drop(point_df.shape[0]-1, inplace=True)
        num_bins = divisors(point_df.shape[0])[-2]
    # we make a scipy stats rv because its the easiest to sample from
    hist = point_table_to_hist(point_df, num_bins=num_bins)
    hist_dist = sp.rv_histogram(hist)
    # we sample 1k variates to see which ones are the best.
    samples = hist_dist.rvs(size=10000)
    hist_trace = go.Histogram(x=samples, nbinsx=25,
                          name='Input Distribution', opacity=0.6, histnorm='probability density')

    # @TODO: let yourself choose how many samples you want AND which distributions to test.
    f = fitter.Fitter(samples, bins=20,
                      distributions=fitter.get_common_distributions())
    f.fit()
    score_df = f.summary()
    png = f.plot_pdf()
    plt.savefig('test.png')
    # CREATE PLOT
    pdf_dict = {}
    X = np.linspace(np.min(hist_dist.rvs(size=1000))-25,
                    np.max(hist_dist.rvs(size=1000))+25, 25)

    for distro in score_df.index:
        sp_dist = eval('scipy.stats.'+distro)
        params = f.fitted_param[distro]
        samples = sp_dist.pdf(X, *params)
        pdf_dict[distro] = (samples)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(hist_trace)

    for pdf in pdf_dict.keys():
        trace = go.Scatter(x=X, y=pdf_dict[pdf], name=f'{pdf} best fit',line=dict(
                    width=2
                ))
        fig.add_trace(trace, secondary_y=True)
    fig.update_layout(bargap=0.01)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
# First route : Render the initial drawing template
@app.route('/')
def home():
    return render_template('draw.html')


# Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    global graph
    if request.method == 'POST':
        final_pred = None
        # Access the image
        draw = request.form['url']
        # Removing the useless part of the url.
        draw = draw[init_Base64:]
        # Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        # Launch prediction
        print('processing image')
        point_df = process_image(image)

        json = predict_image(point_df)
    return render_template('results.html', graphJSON=json)


if __name__ == '__main__':
    app.run(debug=True)

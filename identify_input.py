import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils.Logger import reproduc
from utils.Identify import InputCombination

def main():
    model = RandomForestRegressor(random_state=42)
    model.fit(X, Y)

    explainer = shap.TreeExplainer(model)
    test = X[np.random.choice(X.shape[0], min(200,X.shape[0]), replace=False)]
    shap_values = explainer.shap_values(test)

    sorted_pairs = sorted(zip(abs(shap_values).mean(0).tolist(), input_list), key=lambda x: x[0], reverse=True)
    sorted_input_list = [item[1] for item in sorted_pairs]
    print(sorted_input_list)

    shap.summary_plot(shap_values, test, feature_names=input_list, plot_type="bar", show=True)
    shap.summary_plot(shap_values, test, feature_names=input_list, show=True)

if __name__=="__main__":
    reproduc()
    csv_path = 'dataset/solar.csv'

    input_list = ['Diameter', 'Density', 'Gravity', 'Escape Velocity', 'Rotation Period', 'Length of Day',
        'Distance from Sun', 'Perihelion', 'Aphelion', 'Orbital Period', 'Orbital Velocity']
    output_list = ['Mass']
    D = [[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1., -3.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  1.],
        [ 0.,  0., -2., -1.,  1.,  1.,  0.,  0.,  0.,  1., -1.]]

    X, Y, input_list, units = InputCombination(csv_path, input_list, output_list, D, origin=True, mixed=True, only_binary=True)

    main()

import numpy as np
import matplotlib.pyplot as plt


transfer_metric = {}


# transfer_metric['GP-MPC'] = {'rmse': np.array([0.07240253210609013, 
#                                                0.06001951283653607,
#                                               0.049872511450839645,
#                                                0.043216195868052004,
#                                                0.03819312862247324,
#                                                0.03440810993189095,
#                                               0.031425261835490235]), 
#                              'rmse_std': np.array([0.005935227930378862,
#                                                    0.005650813796205759,
#                                                   0.005132586426383544,
#                                                    0.005048544039303885,
#                                                    0.004922580428116054,
#                                                    0.004879404518445507,
#                                                   0.004910546198568348])}
# transfer_metric['Nonlinear MPC'] = {'rmse': np.array([0.06669989755434953,
#                                             0.053871034117811606,
#                                                   0.04421752503119518,
#                                             0.03683269097948807,
#                                             0.030699167806897958,
#                                             0.026301931256398743,
#                                                   0.023313325828377546]), 
#                                  'rmse_std': np.array([0.0059149924488957496,
#                                                        0.005462706718375091,
#                                                       0.005051390779798208,
#                                                        0.004678658729904046,
#                                                        0.004937412985172684,
#                                                        0.004577798254254554,
#                                                       0.0059149924488957496])}
transfer_metric['Linear MPC'] = {'rmse': np.array([0.10363015782869267,
                                                #    0.08321171658485764,
                                                   0.06556989411791102,
                                                #    0.05575281010753317,
                                                   0.04714083998242881,
                                                #    0.04068921844895369,
                                                   0.03574250675445726]), 
                                 'rmse_std': np.array([0.004825855269260951,
                                                    #    0.004456539838199912,
                                               0.0035402446045955343,
                                                    #    0.0033564715382322897,
                                                       0.0030013781990673905,
                                                    #    0.00278069835757052,
                                               0.002647060557505466])}
transfer_metric['Linear MPC acados'] = {'rmse': np.array([0.09475888182425916,
                                                          0.06284182159429033,
                                                          0.04495251910240445,
                                                          0.034306490036127714,]),
                                        'rmse_std': np.array([0.007239342361923242,
                                                              0.005862348764648783,
                                                              0.004953459868638366,
                                                              0.0042428141668883965,])}

transfer_metric['Nonlinear MPC'] = {'rmse': np.array([0.06669989755434953,
                                            # 0.053871034117811606,
                                                  0.04421752503119518,
                                            # 0.03683269097948807,
                                            0.030699167806897958,
                                            # 0.026301931256398743,
                                                  0.023313325828377546]), 
                                 'rmse_std': np.array([0.0059149924488957496,
                                                    #    0.005462706718375091,
                                                      0.005051390779798208,
                                                    #    0.004678658729904046,
                                                       0.004937412985172684,
                                                    #    0.004577798254254554,
                                                      0.0059149924488957496])}

fig = plt.figure(figsize=(8, 3))
plot_list = [
    'Linear MPC', 'Linear MPC acados', 'Nonlinear MPC'
]
plot_colors = {
    'Linear MPC': 'tab:green',
    'Linear MPC acados': 'skyblue',
    'Nonlinear MPC': 'tab:blue',
}

episode_len_list = [9, 11, 13, 15]
for method in plot_list:
    print(method)
    if method == 'DPPO':
        plt.plot(episode_len_list, transfer_metric[method]['rmse'], label=method, linestyle='--', color=plot_colors[method])
        plt.fill_between(episode_len_list, 
                     transfer_metric[method]['rmse']-transfer_metric[method]['rmse_std'],  
                     transfer_metric[method]['rmse']+transfer_metric[method]['rmse_std'], color=plot_colors[method], alpha=0.1)
    else:
        plt.plot(episode_len_list, transfer_metric[method]['rmse'], label=method, color=plot_colors[method])
        plt.fill_between(episode_len_list, 
                         transfer_metric[method]['rmse']-transfer_metric[method]['rmse_std'],  
                         transfer_metric[method]['rmse']+transfer_metric[method]['rmse_std'], color=plot_colors[method], alpha=0.1)
plt.axvline(x=11, linestyle='-.', color='gray')
plt.text(10.9, 0.12, 'Nominal Task')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0,0.14)
#plt.xlim(0,50)
plt.gca().invert_xaxis()
plt.xlabel("figure 8 trajectory period (s)")
plt.ylabel("rmse")
plt.savefig("generalization_curve.pdf",bbox_inches="tight", pad_inches=0.1)

# transfer_metric['Linear MPC acados'] = {'rmse': np.array([0.10363015782869267,
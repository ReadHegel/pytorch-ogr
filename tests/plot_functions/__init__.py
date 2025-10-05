from .hessian_plot import HessianPlot, InvHessianPlot
from .metadata import Plot, PlotList, Result
from .plot_traces import TracePlot

_plot_list: list[Plot] = [TracePlot, HessianPlot, InvHessianPlot]
plot_list: PlotList = PlotList(_plot_list)

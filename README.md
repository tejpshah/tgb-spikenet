# Evaluating SpikeNet on the Temporal Graph Benchmark  
We evaluated SpikeNet, a new biologically inspired temporal graph representation learning technique, on the recent Temporal Graph Benchmark. For the Node Classification task on $\texttt{tgbn-trade}$, SpikeNet is the worst performing technique, but is similarly competitive to other technical temporal graph representation learning techniques, suggesting model performance may be closely related to the dataset size regime: for small data regimes, the simplest model is preferred. For the Node Classification task on $\texttt{tgbn-genre}$, SpikeNet is the worst-performing technique among all the techniques. 

$$

\begin{table}[!h]
\centering
\begin{tabular}{|l|l|l|}
\hline
\textbf{Method} & \textbf{Test NDCG@10} & \textbf{Val NDCG@10} \\ \hline
Persistent Forecast & 0.855 & 0.860 \\ \hline
Moving Average & 0.823 & 0.841 \\ \hline
DyGFormer & 0.388 & 0.408 \\ \hline
TGN & 0.374 & 0.395 \\ \hline
DyRep & 0.374 & 0.394 \\ \hline \hline 
SpikeNet & 0.334 & 0.380 \\ \hline
\end{tabular}
\caption{\texttt{tgbn-trade} NDCG@10 Leaderboard}
\label{tab:ndcg_scores}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{|l|l|l|}
\hline
\textbf{Method} & \textbf{Test NDCG@10} & \textbf{Val NDCG@10} \\ \hline
Persistent Forecast & 0.509 & 0.499 \\ \hline
Moving Average & 0.367 & 0.403 \\ \hline
DyGFormer & 0.365 & 0.371 \\ \hline
TGN & 0.357 & 0.350 \\ \hline
DyRep & 0.351 & 0.357 \\ \hline \hline 
SpikeNet & $<$ 0.01 & $<$ 0.01 \\ \hline
\end{tabular}
\caption{\texttt{tgbn-genre} NDCG@10 Leaderboard}
\label{tab:genre_ndcg_scores}
\end{table}
$$


# References
- SpikeNet Implementation: https://github.com/EdisonLeeeee/SpikeNet?tab=readme-ov-file
- TGB Implementation: https://github.com/shenyangHuang/TGB/tree/main
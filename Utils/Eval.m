function Eval(Model, EvalD)
    %% Evaluation function

    Performance = table('Size', [12 2], ...
                    'VariableTypes', {'double', 'double'}, ...
                    'VariableNames', ["CARAR-LF", "CARAR"], ...
                    'RowNames', {'AUC';'AP';'AP@5';'PREC@5';'REC@5';'F1@5';'NDCG@5';'AP@10';'PREC@10';'REC@10';'F1@10';'NDCG@10'});
    Result = struct2cell(RankEval(gather(Model.Predict(Model, EvalD, Model.Pm.phi))', gather(EvalD.E'), 5));
    Performance(1:7, 1) = Result(1:7);
    Result = struct2cell(RankEval(gather(Model.Predict(Model, EvalD, Model.Pm.phi))', gather(EvalD.E'), 10));
    Performance(8:12, 1) = Result(3:7);
    Result = struct2cell(RankEval(gather(Model.Adjust(Model, EvalD, Model.Pm.phi))', gather(EvalD.E'), 5));
    Performance(1:7, 2) = Result(1:7);
    Result = struct2cell(RankEval(gather(Model.Adjust(Model, EvalD, Model.Pm.phi))', gather(EvalD.E'), 10));
    Performance(8:12, 2) = Result(3:7);
    
    Performance

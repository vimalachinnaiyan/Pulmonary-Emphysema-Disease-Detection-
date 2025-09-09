function [Fitness] = Objfun_Cls(Soln)

global Feat Targets Bestsol Train_Data Test_Data Train_Target Test_Target

for i = 1:size(Soln, 1)
     Bestsol = Soln(i, :);
     Learnperc = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85];
     for learn = 1:length(Learnperc)
         learnperc = round(size(Feat, 1) * 0.75);
         Train_Data = Feat(1:learnperc, :);
         Train_Target = Targets(1:learnperc);
         Test_Data = Feat(learnperc + 1:end, :);
         Test_Target = Targets(learnperc + 1:end);
         [Eval] = Model_PROPOSED();
     end
     Fitness(i) = (1 / Eval(4)+ Eval(7)+ Eval(10));
end

end
function RescorlaWagnerPlotter(TimeVec, Paradigm, ParadigmStruct)

figure; hold on; 
subplot(3,1,1); plot(TimeVec, Paradigm(1,:),'k', 'LineWidth', 2); 
hold on; axis([-20 200 -1.1 1.1]); box off; ylabel('u_1', 'FontSize', 20);
subplot(3,1,1); hold on; plot(TimeVec,ParadigmStruct.AssociationVec(1,:),'b*');

subplot(3,1,2); plot(TimeVec, Paradigm(2,:),'k', 'LineWidth', 2); 
hold on; axis([-20 200 -1.1 1.1]); box off; ylabel('u_2', 'FontSize', 20);
subplot(3,1,2); hold on; plot(TimeVec,ParadigmStruct.AssociationVec(2,:),'b*');

subplot(3,1,3); plot(TimeVec, Paradigm(3,:),'k', 'LineWidth', 2);
hold on; axis([-20 200 -0.1 1.1]); box off; ylabel('r', 'FontSize', 20);

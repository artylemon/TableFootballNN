li = linspace(0,205,20);
si = linspace(35,305,20);


[LI,SI] = meshgrid(li,si);

li = reshape(LI,[],1);
si = reshape(SI,[],1);

di = sqrt( (354-li).^2 + (166.5-si).^2);
ai = atand( (166.5-si)./(354-li) )./1.8;

lo = li*0.9928+78.112;
so = si*1.0533+30.775;
ao = round(ai*1.0016+0.1673);
po = di*0.0028+2.286;

fake_data = table(li,si,lo,so,ao,po);

%DATA SKEWING:
% for k=1:height(fake_data)
%     if fake_data.li(k)<150 && fake_data.si(k)>175 && fake_data.si(k)<325
%         fake_data.po(k) = fake_data.po(k)*1.1;
%     end
% end

plot3(li,si,ao,'ko')

writetable(fake_data, 'fake_data_skewed.csv')
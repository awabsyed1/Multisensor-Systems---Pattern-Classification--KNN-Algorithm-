function f_1 = filter_extract(B,A,xn,f)
for i = 1:50 
    y(:,i) = filter(B,A,xn(:,i));
    psd = pwelch(y(:,i),[],[],[],f);
    f_1(:,i) = norm(psd)/sqrt(max(size(psd)));
end
end
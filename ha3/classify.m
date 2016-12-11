function z = classify(y,mu,pc,sigma2)

a = (y - mu).^2;
thresh = (2*sigma2) - log((2*pi*sigma2)*(1-pc)/pc);
z = (a >= thresh);

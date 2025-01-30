SELECT count(*) ,"year", flight FROM public.sgl_train_master group by "year", flight
LIMIT 100

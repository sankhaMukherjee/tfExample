

grantPermissions:
	chmod 766 ../bin/*

run:
	../bin/run.sh

runA:
	../bin/run.sh -a

# This is a quick check for timming.
# For thorough results, use the profile
# option
runT:
	echo "Simple timing analysis"
	../bin/run.sh -t

clean:
	../bin/clean.sh

# ---- Easy way of checking the functions ---------
checkLog:
	../bin/checkLog.sh

checkTime:
	../bin/checkTime.sh

test:
	python3 -m pytest ../tests

doc:
	@echo 'Generating a fresh copy of the docs ...'
	@rm -rf ../docs
	@mkdir ../docs
	@sphinx-apidoc -F -o ../docs .
	@sed "s/^extensions.*/& 'sphinx.ext.napoleon',/"        ../docs/conf.py   > ../docs/conf2.py
	@sed "s/^# import/import/"                              ../docs/conf2.py  > ../docs/conf3.py
	@sed "s/^# sys.path./sys.path./"                        ../docs/conf3.py  > ../docs/conf4.py
	@echo '# napoleon extensions over here '                >> ../docs/conf1.py
	@echo '# ------------------------------'                >> ../docs/conf1.py
	@echo 'napoleon_google_docstring = True'                >> ../docs/conf1.py
	@echo 'napoleon_numpy_docstring = True'                 >> ../docs/conf1.py
	@echo 'napoleon_include_init_with_doc = False'          >> ../docs/conf1.py
	@echo 'napoleon_include_private_with_doc = False'       >> ../docs/conf1.py
	@echo 'napoleon_include_special_with_doc = True'        >> ../docs/conf1.py
	@echo 'napoleon_use_admonition_for_examples = False'    >> ../docs/conf1.py
	@echo 'napoleon_use_admonition_for_notes = False'       >> ../docs/conf1.py
	@echo 'napoleon_use_admonition_for_references = False'  >> ../docs/conf1.py
	@echo 'napoleon_use_ivar = False'                       >> ../docs/conf1.py
	@echo 'napoleon_use_param = True'                       >> ../docs/conf1.py
	@echo 'napoleon_use_rtype = True'                       >> ../docs/conf1.py
	@echo ''                                                >> ../docs/conf1.py
	@echo ''                                                >> ../docs/conf1.py
	@cat ../docs/conf4.py ../docs/conf1.py > ../docs/conf5.py
	@rm ../docs/conf.py ../docs/conf1.py ../docs/conf2.py ../docs/conf3.py ../docs/conf4.py
	@mv ../docs/conf5.py ../docs/conf.py
	@echo 'Generating the doc tree ....'
	@echo '----------------------------'
	sphinx-build -b html -aE -d ../docs/doctrees -c ../docs ../docs ../docs/_build/html


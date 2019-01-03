
install:
	@pip install -q -r requirements.txt

note:
	@jupyter notebook

web:
	@python app.py

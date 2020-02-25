from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class MyForm(FlaskForm):
    class Meta:  # Ignoring CSRF security feature.
        csrf = False
    input_field1 = StringField(label='input:', id='inf_user1',
                              validators=[DataRequired()])
    input_field2 = StringField(label='input:', id='inf_user1',
                              validators=[DataRequired()])
    submit = SubmitField('Submit')
    common_bots = []



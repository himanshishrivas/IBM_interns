# Generated by Django 2.2.6 on 2019-10-27 02:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('patientmedicalhistory', '0004_medhistoryinfo_med_history_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='medhistoryinfo',
            name='type',
            field=models.CharField(default='symptoms', max_length=100),
        ),
    ]
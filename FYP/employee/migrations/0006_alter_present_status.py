# Generated by Django 4.2.16 on 2025-01-15 09:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('employee', '0005_alter_present_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='present',
            name='status',
            field=models.CharField(choices=[('P', 'Present'), ('A', 'Absent'), ('L', 'Late'), ('E', 'Excused')], default='A', max_length=1),
        ),
    ]

# Generated by Django 4.2.16 on 2025-02-19 02:11

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('employee', '0013_alter_shift_date'),
    ]

    operations = [
        migrations.CreateModel(
            name='Event',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('description', models.TextField()),
                ('start_time', models.DateTimeField()),
                ('end_time', models.DateTimeField()),
            ],
        ),
        migrations.CreateModel(
            name='ShiftCalendar',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('shift_type', models.CharField(choices=[('A', 'Shift A (04:00 - 12:00)'), ('B', 'Shift B (12:00 - 20:00)'), ('C', 'Shift C (20:00 - 04:00)'), ('D', 'Day Off')], max_length=1)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'unique_together': {('user', 'date')},
            },
        ),
        migrations.DeleteModel(
            name='Events',
        ),
        migrations.RemoveField(
            model_name='shift',
            name='date',
        ),
        migrations.AlterField(
            model_name='shift',
            name='shift_type',
            field=models.CharField(choices=[('A', 'Shift A (04:00 - 12:00)'), ('B', 'Shift B (12:00 - 20:00)'), ('C', 'Shift C (20:00 - 04:00)'), ('D', 'Day Off')], max_length=1),
        ),
    ]

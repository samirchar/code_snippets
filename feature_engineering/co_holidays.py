"""This module generates features based on Holidays (or especial dates) in Colombia
"""
from datetime import date
from workalendar.america import Colombia
from workalendar.core import Calendar, SUN, SAT

cal = Colombia()
current_date = date.today()


class Holidays:
    """Class that makes calculations based on Holidays
    
    :return: class: `~Holidays`
    """
    def __init__(self):
        """init
        """
        
        self.holidays = []
        self.loaded_years = []

    def add_year(self, year):
        """Given a year, this method appends all Holidays
        for that year to the list attribute 'holidays'.
        After loading, it appends the year number to the 'loaded_years'
        list attribute in order to keep a state of what years are already loaded.
        
        :param year: the year we want to load
        :type year: int
        """

        self.holidays.extend(cal.holidays(year))

        # Mothers day: 2nd Sunday of May.
        self.holidays.append(
            (Calendar.get_nth_weekday_in_month(year, 5, SUN, 2), 'Mothers day'))
        # Fathers day: 3rd Sunday of June
        self.holidays.append(
            (Calendar.get_nth_weekday_in_month(year, 6, SUN, 3), 'Mothers day'))
        # Halloween: Oct 31
        self.holidays.append((date(year, 10, 31), 'Halloween'))
        # Velitas 7 Dic
        self.holidays.append((date(year, 12, 7), 'Día de las Velitas'))
        # Amor y amistad
        self.holidays.append(
            (Calendar.get_nth_weekday_in_month(year, 9, SAT, 3), 'Amor y amistad'))
        # San valentín
        self.holidays.append((date(year, 2, 14), 'Día de San Valentín'))
        # Día del Hombre
        self.holidays.append((date(year, 3, 19), 'Día del Hombre'))
        # Día de la Mujer
        self.holidays.append((date(year, 3, 8), 'Día de la Mujer'))

        self.loaded_years.append(year)

    def is_date_loaded(self, date_obj):
        """Checks if the info needed for a current date is loaded,
        if not, it loads it
        
        :param date_obj: datetime object that we need to make a calculation
        :type date_obj: datetime
        """
        if not date_obj.year in self.loaded_years:
            self.add_year(date_obj.year)
        else:
            pass

    def get_holidays_between_dates(self, start, end):
        """Gets the holidays between two dates.
        
        :param start: start date
        :type start: datetime
        :param end: end date
        :type end: datetime
        :return: the list of holidays that fall between start and end date
        :rtype: list
        """
        self.is_date_loaded(start)
        self.is_date_loaded(end)

        for i in range(start.year+1, end.year):
            self.is_date_loaded(date(i, 1, 1))

        return [d[0] for d in self.holidays if (d[0] >= start and d[0] <= end)]

    def count_holidays_between_dates(self, start, end):
        """Counts the holidays between two dates.
        
        :param start: start date
        :type start: datetime
        :param end: end date
        :type end: datetime
        :return: the number of holidays that fall within start and end date
        :rtype: int
        """
        return len(self.get_holidays_between_dates(start, end))

    def days_until_next_holiday(self, date_obj=current_date):
        """Given a date, this method counts the number of days until the next holiday
        
        :param date_obj: the date we want to evaluate, by default current_date
        :type date_obj: datetime, required
        :return: the number of days until de next holiday
        :rtype: int
        """
        self.is_date_loaded(date_obj)
        holiday_dates = [d[0] for d in self.holidays]

        # Handle end of dicembre
        if (date_obj >= date(date_obj.year, 12, 25)) and date_obj <= date(date_obj.year, 12, 31):
            return (date(date_obj.year+1, 1, 1)-date_obj).days
        else:
            return min([(i - date_obj).days for i in holiday_dates if i > date_obj])

    def days_after_last_holiday(self, date_obj=current_date):
        """Given a date, this method counts the number of days since the last holiday
        
        :param date_obj: the date we want to evaluate, defaults to current_date
        :type date_obj: datetime, required
        :return: the number of days since the last holiday
        :rtype: int
        """
        self.is_date_loaded(date_obj)
        holiday_dates = [d[0] for d in self.holidays]

        # Handle end of dicembre
        if date_obj == date(date_obj.year, 1, 1):
            return (date_obj - date(date_obj.year-1, 12, 25)).days
        else:
            return min([(date_obj-i).days for i in holiday_dates if i < date_obj])

class House:
    def __init__(self, location, house_type, deal_type, price, completion_year) -> None:
        self.location = location
        self.house_type = house_type
        self.deal_type = deal_type
        self.price = price
        self.completion_year = completion_year

    def ShowDetail(self):
        if type(self.price) == list:
            price = "{deposit}/{monthlyRent}".format(monthlyRent = self.price[1], deposit = self.price[0])
        else:
            price = str(self.price)

        print(self.location, self.house_type, self.deal_type, price, "{}년".format(self.completion_year))

houses = [House("강남", "아파트", "매매", "10억", 2010) ,]
houses.append(House("마포", "오피스텔", "전세", "5억", 2007))
houses.append(House("송파", "빌라", "월세", [500,50], 2000))

for i in houses:
    i.ShowDetail()
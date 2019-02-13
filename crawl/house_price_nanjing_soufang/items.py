# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class community_item(scrapy.Item):
	# define the fields for your item here like:
	# name = scrapy.Field()
	code = scrapy.Field()
	name = scrapy.Field()
	address = scrapy.Field()
	longitude = scrapy.Field()
	latitude = scrapy.Field()
	district = scrapy.Field()
	region = scrapy.Field()
	price_time_series = scrapy.Field()
	architectural_age = scrapy.Field()
	construction_area = scrapy.Field()
	floor_area = scrapy.Field()
	volume_rate = scrapy.Field()
	greening_rate = scrapy.Field()
	property_costs = scrapy.Field()
	pass

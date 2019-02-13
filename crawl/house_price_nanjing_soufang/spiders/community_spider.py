import scrapy
import re
import json
from scrapy.http.request import Request
#from house_price_nanjing_soufang.items import community_item

price_time_series_url_prefix = r'http://fangjia.fang.com/pinggu/ajax/ChartAjaxContainMax.aspx?dataType=proj&city=%u5357%u4EAC&KeyWord='
price_time_series_url_suffix = r'&year=2'
geocoding_url_prefix = r'http://restapi.amap.com/v3/geocode/geo?key=fa2218a86058bd37976ca57be4310710&city=南京&batch=true&address='
GCJ2WGS_url_prefix = r'http://api.zdoz.net/transmore.ashx?lngs='
GCJ2WGS_url_mid = r'&lats='
GCJ2WGS_url_suffix = r'&type=2'

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

class CommunitySpider(scrapy.Spider):
	name = 'community'
	
	start_urls = [
		'http://esf.nanjing.fang.com/housing/265__1_0_0_0_1_0_0_0/', #gulou district
		'http://esf.nanjing.fang.com/housing/268__1_0_0_0_1_0_0_0/', #jiangning district
		'http://esf.nanjing.fang.com/housing/270__1_0_0_0_1_0_0_0/', #pukou district
		'http://esf.nanjing.fang.com/housing/264__1_0_0_0_1_0_0_0/', #xuanwu district
		'http://esf.nanjing.fang.com/housing/267__1_0_0_0_1_0_0_0/', #jianye district
		'http://esf.nanjing.fang.com/housing/271__1_0_0_0_1_0_0_0/', #qixia district
		'http://esf.nanjing.fang.com/housing/272__1_0_0_0_1_0_0_0/', #yuhua district
		'http://esf.nanjing.fang.com/housing/263__1_0_0_0_1_0_0_0/', #qinhuai district
		'http://esf.nanjing.fang.com/housing/269__1_0_0_0_1_0_0_0/', #liuhe district
		'http://esf.nanjing.fang.com/housing/274__1_0_0_0_1_0_0_0/', #lishui district
		'http://esf.nanjing.fang.com/housing/275__1_0_0_0_1_0_0_0/', #gaochun district
	]
    
	def parse(self, response):
		# follow pagination links
		for href in response.xpath('//div[@class="fanye gray6"]/a[@id="PageControl1_hlk_next"]/@href'):
			yield response.follow(href, self.parse)
            
		# follow links to community pages
		for href in response.xpath('//dl[@class="plotListwrap clearfix"]/dd/p/a[@class="plotTit"]/@href'):
			yield response.follow(href, self.parse_community_home_page)
            
	def parse_community_home_page(self, response):
		item_home_page = community_item()
		item_home_page['name'] = response.xpath('//div[@class="firstright"]/div[@class="Rbigbt clearfix"]/h1/strong/text()').extract_first()
		item_home_page['code'] = response.xpath('//input[@id="projCode"]/@value').extract_first()

		href = response.xpath('//li[@data="xqxq"]/a/@href').extract_first()
		yield Request(url = href, callback = self.parse_community_details, meta={'item': item_home_page})

	def parse_community_details(self, response):
		item_details = response.meta['item']

		lines = response.xpath('//dl[@class=" clearfix mr30"]/dd')
		for line in lines:
			tag = line.xpath('.//strong/text()').extract_first()
			if tag.find(u"小区地址") != -1:
				item_details['address'] = line.xpath('./text()').extract_first()
			elif tag.find(u"所属区域") != -1:
				item_details['district'] = re.split(' ', line.xpath('./text()').extract_first())[0]
				item_details['region'] = re.split(' ', line.xpath('./text()').extract_first())[1]
			elif tag.find(u"建筑年代") != -1:
				item_details['architectural_age'] = re.match('\d{4}', line.xpath('./text()').extract_first()).group(0)
			elif tag.find(u"建筑面积") != -1:
				item_details['construction_area'] = re.match('\d+', line.xpath('./text()').extract_first()).group(0)
			elif tag.find(u"占地面积") != -1:
				item_details['floor_area'] = re.match('\d+', line.xpath('./text()').extract_first()).group(0)
			elif tag.find(u"容 积 率") != -1:
				item_details['volume_rate'] = re.match('\d+.\d+', line.xpath('./text()').extract_first()).group(0)
			elif tag.find(u"绿 化 率") != -1:
				item_details['greening_rate'] = line.xpath('./text()').extract_first()
			elif tag.find(u"物 业 费") != -1:
				item_details['property_costs'] = re.match('\d+.\d+', line.xpath('./text()').extract_first()).group(0)

		href = geocoding_url_prefix + item_details['address']
		yield Request(url = href, callback = self.parse_community_geocodes_GCJ, meta={'item': item_details})

	def parse_community_geocodes_GCJ(self, response):
		item_geocodes_GCJ = response.meta['item']
		js = json.loads(response.text)
		if js['status'] == '1':
			item_geocodes_GCJ['longitude'] = re.split(',', js['geocodes'][0]['location'])[0]
			item_geocodes_GCJ['latitude'] = re.split(',', js['geocodes'][0]['location'])[1]

			href = GCJ2WGS_url_prefix + item_geocodes_GCJ['longitude'] + GCJ2WGS_url_mid + item_geocodes_GCJ['latitude'] + GCJ2WGS_url_suffix
			yield Request(url = href, callback = self.parse_community_geocodes_WGS, meta={'item': item_geocodes_GCJ})

	def parse_community_geocodes_WGS(self, response):
		item_geocodes_WGS = response.meta['item']
		js = json.loads(response.text)
		item_geocodes_WGS['longitude'] = js[0]['Lng']
		item_geocodes_WGS['latitude'] = js[0]['Lat']

		href = price_time_series_url_prefix + item_geocodes_WGS['code'] + price_time_series_url_suffix
		yield Request(url = href, callback = self.parse_community_price_time_series, meta={'item': item_geocodes_WGS})

	def parse_community_price_time_series(self, response):
		item_price_time_series = response.meta['item']
		point_dict = re.split(r'&', response.text)[0][1: -1]
		point_str = point_dict.replace('[', '').replace(']', '')
		axis_value = re.split(",", point_str)
		price = []
		count = 0
		for i in axis_value:
		    count = count + 1
		    if count%2 == 0:
		        price.append(i)
		item_price_time_series['price_time_series'] = price

		yield item_price_time_series
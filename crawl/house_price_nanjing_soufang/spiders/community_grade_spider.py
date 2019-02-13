import scrapy
import re
import json
from scrapy.http.request import Request

class community_grade_item(scrapy.Item):
	# define the fields for your item here like:
	# name = scrapy.Field()
	code = scrapy.Field()
	grade = scrapy.Field()
	pass

class CommunitySpider(scrapy.Spider):
	name = 'community_grade'
	
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
		item_grade = community_grade_item()
		item_grade['code'] = response.xpath('//input[@id="projCode"]/@value').extract_first()
		item_grade['grade'] = 5 - len(response.xpath('//div[@class="xqgrade clearfix"]//div[@class="dj"]/i[@class="no2"]'))
		yield item_grade
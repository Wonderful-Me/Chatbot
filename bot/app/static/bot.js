
var is_chart=0;

var text = $("#f-left");
text.focus();
var d, h, m,
    i = 0;

var dom = document.getElementById('container');
var myChart = echarts.init(dom, null, {
      renderer: 'canvas',
      useDirtyRect: false
    });
var app = {};
    
var option;

   // prettier-ignore
   var my_data={
   dataAxis : [],
    data :[]};
   let yMax = 500;
   let dataShadow = [];
   for (let i = 0; i < my_data.data.length; i++) {
	 dataShadow.push(yMax);
   }
   option = {
	 title: {
	   text: '',
	   subtext: ''
	 },
	 xAxis: {
	   data: [],
	   axisLabel: {
		 inside: true,
		 color: '#fff'
	   },
	   axisTick: {
		 show: false
	   },
	   axisLine: {
		 show: false
	   },
	   z: 10
	 },
	 yAxis: {
	   axisLine: {
		 show: false
	   },
	   axisTick: {
		 show: false
	   },
	   axisLabel: {
		 color: '#999'
	   }
	 },
	 grid:{
		 x:35,
		 y:50,
		 x2:35,
		y2:35,
		borderWidth: 1,
	 },
	 dataZoom: [
	   {
		 type: 'inside'
	   }
	 ],
	 series: [
	   {
		 type: 'bar',
		 showBackground: true,
		 itemStyle: {
		   color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
			 { offset: 0, color: '#83bff6' },
			 { offset: 0.5, color: '#188df0' },
			 { offset: 1, color: '#188df0' }
		   ])
		 },
		 emphasis: {
		   itemStyle: {
			 color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
			   { offset: 0, color: '#2378f7' },
			   { offset: 0.7, color: '#2378f7' },
			   { offset: 1, color: '#83bff6' }
			 ])
		   }
		 },
		 data: []
	   }
	 ]
   };
	function charts(){
		$("#container").remove();
		$("body").append('<div id="container" style="background-color: #FFFFFF"></div>');
		
		  $.ajax({
			url:'/getjson2',
			type:'GET',
			success:function(result){
				
				myChart.hideLoading();
				dom = document.getElementById('container');
				myChart = echarts.init(dom, null, {
				renderer: 'canvas',
				useDirtyRect: false
		  		});
				  for (let i = 0; i < result.dataAxis.length; i++) {
					dataShadow.push(yMax);
				  };
			   	// 填入数据
				myChart.setOption( {
					title: {
					   text: '实时新冠',
	   					subtext: '各个省市确诊人数'
					},
					xAxis: {
					  data: result.dataAxis,
					  axisLabel: {
						inside: true,
						color: '#fff'
					  },  
					  axisTick: {
						show: false
					  },
					  axisLine: {
						show: false
					  },
					  z: 10
					},
					yAxis: {
					  axisLine: {
						show: false
					  },
					  axisTick: {
						show: false
					  },
					  axisLabel: {
						color: '#999'
					  }
					},
					grid:{
						x:35,
						y:50,
						x2:35,
					   y2:35,
					   borderWidth: 1,
					},
					dataZoom: [
					  {
						type: 'inside'
					  }
					],
					series: [
					  {
						type: 'bar',
						showBackground: true,
						itemStyle: {
						  color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
							{ offset: 0, color: '#83bff6' },
							{ offset: 0.5, color: '#188df0' },
							{ offset: 1, color: '#188df0' }
						  ])
						},
						emphasis: {
						  itemStyle: {
							color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
							  { offset: 0, color: '#2378f7' },
							  { offset: 0.7, color: '#2378f7' },
							  { offset: 1, color: '#83bff6' }
							])
						  }
						},
						data: result.data
					  }
					]
				  });
				}})
	}
	$("#btn2").click(function()
	{
		if(is_chart==1)
		window.location.href="https://voice.baidu.com/act/newpneumonia/newpneumonia/?from=osari_aladin_banner";
		else
		location.reload([bForceGet]); 
	});
	

	for (let i = 0; i < my_data.data.length; i++) {
	  dataShadow.push(yMax);
	}

	// Enable data zoom when user click bar.
	const zoomSize = 6;
	myChart.on('click', function (params) {
	  console.log(dataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)]);
	  myChart.dispatchAction({
		type: 'dataZoom',
		startValue: dataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)],
		endValue:
		  dataAxis[Math.min(params.dataIndex + zoomSize / 2, data.length - 1)]
	  });
	});
	
		if (option && typeof option === 'object') {
		  myChart.setOption(option);
		  
		}
	
		window.addEventListener('resize', myChart.resize);



function action_chatbot() 
{
	if(text.val()==null||text.val()=="")
	{
		text.focus();
		return;
	}

	$(".b-body").append("<div class='mWord'><span><img  src=\"https://user-images.githubusercontent.com/74918703/169009555-8fcd536b-248f-4372-a737-ff0abcbec25b.png\" width=40 height=40/></span><p>" + text.val() + "</p></div>");
	$(".b-body").scrollTop(10000000);
	$(".b-body").append("<div class='wait'><span><img  src=\"https://user-images.githubusercontent.com/74918703/169009681-a57e3c8d-66e4-40fa-b879-fffe9106c113.png\" width=40 height=40/></span> <p></p><div class='Ball'></div><div class='Ball'></div><div class='Ball'></div></div>");
	$(".b-body").scrollTop(10000000);
	var args_post=
	{
        url: "/post_text",
        type: "POST",
		data: {'mytext':text.val()},
        success: function (data) {
            console.log(data);
        }
        }
	
		var args_chart= {
		
			url: "/get_chart",
			type: "GET",
			success:function(result)
			{
				if(result[0]=='1')
				{
					charts();
					myChart.showLoading();
					is_chart=1;
				}
				else{
					is_chart=0;
					$("#container").remove();
					$("body").append('<div id="container" ></div>');
					dom = document.getElementById('container');
					myChart = echarts.init(dom, null, {
					renderer: 'canvas',
					useDirtyRect: false
		  		});
				  for (let i = 0; i < result.dataAxis.length; i++) {
					dataShadow.push(yMax);
				  };
				myChart.setOption(option);
				
				}
				
			}

		}

	var args_get= {
		
			url: "/get_text",
			type: "GET",
			success:function(result)
			{
				$('.wait').remove();
				setDate();
				if(result[0]=='<')
				$(".b-body").append("<div class='rotWord'><span><img alt=\"\" src=\"https://user-images.githubusercontent.com/74918703/169009681-a57e3c8d-66e4-40fa-b879-fffe9106c113.png\" width=40 height=40/></span> " + result + "</div>");
				else
				$(".b-body").append("<div class='rotWord'><span><img alt=\"\" src=\"https://user-images.githubusercontent.com/74918703/169009681-a57e3c8d-66e4-40fa-b879-fffe9106c113.png\" width=40 height=40/></span> <a id='member'>" + result + "</a></div>");

				$(".b-body").scrollTop(10000000);
			}

		}
	
	$.ajax(args_post);
	$.ajax(args_get);
	$.ajax(args_chart);
	text.val("");
	text.focus();
	
};


function setDate(){
	d = new Date();
	if (m != d.getMinutes()) {
	  m = d.getMinutes();
	  $(".b-body").append('<div class="timestamp">' +'-------'+ d.getHours() + ':' + m +'-------'+ '</div>');
	}
  }
  

  $("#btn").click(function()
  {
	  action_chatbot();
  });


$(document).keydown(function(event)
{
	if(event.keyCode==13)
	{
		//action();
		action_chatbot();
	}
});

function ajax(mJson)
{
	var type=mJson.type||'get';
	var url=mJson.url;
	var data=mJson.data;
	var success=mJson.success;
	var error=mJson.error;
	var dataStr='';
	
	if(data)
	{
		var arr = Object.keys(data);
		var len = arr.length;
		var i = 0;
		
		for (var key in data)
		{
			dataStr+=key+'='+data[key];
	
			if (++i<len)
			{
				dataStr+='&';
			}
		}
		
		if(type.toLowerCase()=='get')
		{
			url+='?'+dataStr;
		}
	}
	
	console.log(url);
	
	var xhr=new XMLHttpRequest();
	xhr.open(type,url,true);
	xhr.setRequestHeader('content-type' , 'application/x-www-form-urlencoded');
	xhr.send(null);

	xhr.onreadystatechange=function()
	{
		if(xhr.readyState==4)
		{
			if(xhr.status>=200&&xhr.status<300)
			{
				success&&success(xhr.responseText);
			}
			else
			{
				error&&error(xhr.status);
			}
		}
	}
}		

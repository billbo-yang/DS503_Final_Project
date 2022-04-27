# imports
import sys
import numpy as np
import joblib
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def display_help():
    print("Usage Information:")
    print("python3 life_expectancy_predict.py [filename]\n")
    print("The file should include lines of input data like the following:")
    print("Country,Year,Status,Adult_Mortality,Infant_Deaths,Alcohol,"\
    "Percentage_Expenditure,Hepatitis_B,Under_Five_Deaths,Polio,"\
    "Total_Expenditure,Diptheria,HIV/AIDS,GDP,Population,Thinness_1-19_Years,"\
    "Thinness_5-9_Years,Income_Composition_of_Resources,Schooling")
    print("Where:")
    print("Country is the string name of the country")
    print("Year is the year that the data is from")
    print("Status is the development status of the country (Developed/Developing)")
    print("Adult Mortality is the adult mortality rates of both sexes (probability"\
        " of dying between 15 and 60 years per 1000 population)")
    print("Infant Deaths is the number of infant deaths per 1000 population")
    print("Alcohol is the alcohol, recorded per capital (15+) consumption in litres pure alchol")
    print("Percentage Expenditure is the country's expenditure on health as a"\
        " percentage of its GDP per capita")
    print("Hepatitis B is the Hepatitis B percentage immunization coverage among"\
        " 1 year olds")
    print("Measles is the number of reported measles cases per 1000 population")
    print("Under Five Deaths is the number of under-five deaths per 1000 population")
    print("Polio is the polio immunization coverage among 1 year olds")
    print("Total Expenditure is the general government expenditure on health"\
        " as a percentage of total government expenditure")
    print("Diptheria is the percentage diptheria tetanus toxoid and pertussis immunization"\
        " coverage among 1 year olds")
    print("HIV/AIDS is teh deaths per 1000 live births to HIV/AIDS for 0-4 year olds")
    print("GDP is the gross domestic product per capita in USD")
    print("Population is the population of the country")
    print("Thinness 1-19 Years is the prevalence of thinness among children and adolescents"\
        " between children ages 10-19 as a percentage")
    print("Thinness 5-9 Years is the prevalence of thinness among children from ages 5-9 as a"\
        " percentage")
    print("Income Composition of Resources is the country's Human Development INdex in terms of"\
        " income composition of resources (an index ranging from 0 to 1)")
    print("Schooling is the number of years spent in school")

def main(file):
    rf_model = joblib.load("./country_le_rf_model.joblib")
    data = pd.read_csv(file)

    # replace country with numbers
    data['Country'] = data['Country'].replace(['Afghanistan' , 'Albania' , 'Algeria' , 'Angola' , 'Antigua and Barbuda' , 'Argentina' , 'Armenia' , 'Australia' , 'Austria' , 'Azerbaijan' , 'Bahamas' , 'Bahrain' , 'Bangladesh' , 'Barbados' , 'Belarus' , 'Belgium' , 'Belize' , 'Benin' , 'Bhutan' , 'Bolivia (Plurinational State of)' , 'Bosnia and Herzegovina' , 'Botswana' , 'Brazil' , 'Brunei Darussalam' , 'Bulgaria' , 'Burkina Faso' , 'Burundi' , "Côte d'Ivoire" , 'Cabo Verde' , 'Cambodia' , 'Cameroon' , 'Canada' , 'Central African Republic' , 'Chad' , 'Chile' , 'China' , 'Colombia' , 'Comoros' , 'Congo' , 'Costa Rica' , 'Croatia' , 'Cuba' , 'Cyprus' , 'Czechia' , "Democratic People's Republic of Korea" , 'Democratic Republic of the Congo' , 'Denmark' , 'Djibouti' , 'Dominican Republic' , 'Ecuador' , 'Egypt' , 'El Salvador' , 'Equatorial Guinea' , 'Eritrea' , 'Estonia' , 'Ethiopia' , 'Fiji' , 'Finland' , 'France' , 'Gabon' , 'Gambia' , 'Georgia' , 'Germany' , 'Ghana' , 'Greece' , 'Grenada' , 'Guatemala' , 'Guinea' , 'Guinea-Bissau' , 'Guyana' , 'Haiti' , 'Honduras' , 'Hungary' , 'Iceland' , 'India' , 'Indonesia' , 'Iran (Islamic Republic of)' , 'Iraq' , 'Ireland' , 'Israel' , 'Italy' , 'Jamaica' , 'Japan' , 'Jordan' , 'Kazakhstan' , 'Kenya' , 'Kiribati' , 'Kuwait' , 'Kyrgyzstan' , "Lao People's Democratic Republic" , 'Latvia' , 'Lebanon' , 'Lesotho' , 'Liberia' , 'Libya' , 'Lithuania' , 'Luxembourg' , 'Madagascar' , 'Malawi' , 'Malaysia' , 'Maldives' , 'Mali' , 'Malta' , 'Mauritania' , 'Mauritius' , 'Mexico' , 'Micronesia (Federated States of)' , 'Mongolia' , 'Montenegro' , 'Morocco' , 'Mozambique' , 'Myanmar' , 'Namibia' , 'Nepal' , 'Netherlands' , 'New Zealand' , 'Nicaragua' , 'Niger' , 'Nigeria' , 'Norway' , 'Oman' , 'Pakistan' , 'Panama' , 'Papua New Guinea' , 'Paraguay' , 'Peru' , 'Philippines' , 'Poland' , 'Portugal' , 'Qatar' , 'Republic of Korea' , 'Republic of Moldova' , 'Romania' , 'Russian Federation' , 'Rwanda' , 'Saint Lucia' , 'Saint Vincent and the Grenadines' , 'Samoa' , 'Sao Tome and Principe' , 'Saudi Arabia' , 'Senegal' , 'Serbia' , 'Seychelles' , 'Sierra Leone' , 'Singapore' , 'Slovakia' , 'Slovenia' , 'Solomon Islands' , 'Somalia' , 'South Africa' , 'South Sudan' , 'Spain' , 'Sri Lanka' , 'Sudan' , 'Suriname' , 'Swaziland' , 'Sweden' , 'Switzerland' , 'Syrian Arab Republic' , 'Tajikistan' , 'Thailand' , 'The former Yugoslav republic of Macedonia' , 'Timor-Leste' , 'Togo' , 'Tonga' , 'Trinidad and Tobago' , 'Tunisia' , 'Turkey' , 'Turkmenistan' , 'Uganda' , 'Ukraine' , 'United Arab Emirates' , 'United Kingdom of Great Britain and Northern Ireland' , 'United Republic of Tanzania' , 'United States of America' , 'Uruguay' , 'Uzbekistan' , 'Vanuatu' , 'Venezuela (Bolivarian Republic of)' , 'Viet Nam' , 'Yemen' , 'Zambia' , 'Zimbabwe' , 'Cook Islands' , 'Dominica' , 'Marshall Islands' , 'Monaco' , 'Nauru' , 'Niue' , 'Palau' , 'Saint Kitts and Nevis' , 'San Marino' , 'Tuvalu'], [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100 ,101 ,102 ,103 ,104 ,105 ,106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,123 ,124 ,125 ,126 ,127 ,128 ,129 ,130 ,131 ,132 ,133 ,134 ,135 ,136 ,137 ,138 ,139 ,140 ,141 ,142 ,143 ,144 ,145 ,146 ,147 ,148 ,149 ,150 ,151 ,152 ,153 ,154 ,155 ,156 ,157 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,165 ,166 ,167 ,168 ,169 ,170 ,171 ,172 ,173 ,174 ,175 ,176 ,177 ,178 ,179 ,180 ,181 ,182 ,183 ,184 ,185 ,186 ,187 ,188 ,189 ,190 ,191 ,192 ,193])

    # replace development status with numbers
    data['Status'] = data['Status'].replace(['Developing', 'Developed'], [1,2])
    
    pred = rf_model.predict(data)
    print(pred)

    pd.DataFrame(pred).to_csv("{}_pred.csv".format(file), index=False)

if __name__ == "__main__":
    # parse command line arguments
    num_commands = len(sys.argv)
    if num_commands != 2:
        display_help()
        exit(-1)

    data_file = sys.argv[1]

    main(data_file)
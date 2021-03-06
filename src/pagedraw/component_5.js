// Generated by https://pagedraw.io/pages/8950
import React from 'react';
import './component_5.css';
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import { Redirect } from 'react-router'
import axios from 'axios'
// import axios from 'axios';

var url = 'http://localhost:5000';

export default class component5 extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            roomCode: '',
            score: '',
            redirect: false
        }

        this.handleEvent = this.handleEvent.bind(this);
        this.handleClick = this.handleClick.bind(this);
    }

    handleEvent() {
        this.state.roomCode = document.getElementsByClassName('component_2-text_input-6')[0] ? document.getElementsByClassName('component_2-text_input-6')[0].value : ''
        this.state.score = document.getElementsByClassName('component_2-text_input_2')[0] ? document.getElementsByClassName('component_2-text_input_2')[0].value : ''
    }

    handleClick() {
        var room = localStorage.getItem('room')
        var score = localStorage.getItem('score')
        axios.get(url + '/check?room=' + room)
        .then(response => {
            console.log(response)
            console.log(response.data)
            if(response.data == 'true' || response.data == true){
                axios.post(url + '/input',{
                    room: room,
                    score: score
                }).then(response => {
                    response = response.data
                    if(response == 'false' || response == false){
                        this.setState({redirect: 'bamboozle'});
                    } else if (response == 'true' || response == true){
                        this.setState({redirect: 'ayylmao'});
                    } else {
                        this.setState({redirect: 'waiting'});
                    }
                })
            } else if (response.data == 'false' || response.data == false){
                // this.setState({redirect: 'bamboozle'});
            }
        })
    }

    render() {
        const { redirect } = this.state;

        if (redirect == 'bamboozle') {
            return <Redirect to='/bamboozle'/>;
        } else if (redirect == 'ayylmao'){
            return <Redirect to='/ayylmao'/>;
        }


        return (<div className="component_5-component_5">
        <div className="component_5-0">
            <div className="component_5-0-0" /> 
            <div className="component_5-text-3">
                {"Yao's Millionaire "}
            </div>
            <div className="component_5-0-2" /> 
        </div>
        <div className="component_5-1">
            <div className="component_5-1-0" /> 
            <div className="component_5-rectangle-5">
                <div className="component_5-1-1-0">
                    <img src="https://ucarecdn.com/3af61bfd-4bfd-48b6-a761-a873c1572350/" className="component_5-image-6" /> 
                </div>
                <div className="component_5-1-1-1">
                    <div className="component_5-rectangle_4">
                        <div className="component_5-1-1-1-0-0">
                            <div className="component_5-text_4" onClick={this.handleClick}>Check</div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="component_5-1-2" /> 
        </div>
        <div className="component_5-2" /> 
    </div>);
    }
}
